#!/usr/bin/env python3
"""Raspberry Pi NoIR camera server.

Streams the processed preview to a TCP client and accepts remote parameter
commands.  Runs headless by default; pass --gui to show a local OpenCV window.

Remote / local (when --gui is used) key commands:
  w/s or +/- : increase/decrease threshold
  a/z        : increase/decrease Gaussian blur kernel
  d/c        : increase/decrease minimum object area
  i          : toggle 180-degree rotation
  q          : quit

Calibration is sent by the client as a binary frame:
  0x01 | uint32 length | JSON {"calib": [{"x":…,"y":…}, …]}  (4 points, normalised to display panel)
"""

import json
import os
import socket
import struct
import sys
import threading
import time

import time

import numpy as np
from picamera2 import Picamera2
import cv2
from pythonosc import udp_client


FRAME_DURATION_US = 16666  # 60 fps target
EXPOSURE_TIME_US = 30000
ANALOG_GAIN = 40.0
INITIAL_THRESHOLD = 43
THRESHOLD_STEP = 5
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 480
BLUR_KERNEL_INITIAL = 5
BLUR_KERNEL_MIN = 1
BLUR_KERNEL_MAX = 31
BLUR_KERNEL_STEP = 2
MIN_AREA_INITIAL = 500
MIN_AREA_STEP = 100
MAX_OBJECTS = 10

SERVER_HOST = ""
SERVER_PORT = 12345
JPEG_QUALITY = 70

# Stream output defaults (can be overridden via calibration.json)
_DEFAULT_STREAM_FPS = 15.0
_DEFAULT_STREAM_WIDTH = 640
_DEFAULT_STREAM_HEIGHT = 240
_DEFAULT_JPEG_QUALITY = 50

CALIB_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# Default OSC/network settings.  These can be overridden by calibration.json.
_DEFAULT_OSC_IP = "192.168.1.106"
_DEFAULT_OSC_PORT = 9000
_DEFAULT_OSC_QUEUE_FPS = 10.0
_DEFAULT_OSC_CLICK_GAP_MS = 50.0
_DEFAULT_PROJECTION_ASPECT = 1.7777777777777777
_DEFAULT_PERIPHERAL_MARGIN = 0.10
_DEFAULT_PERIPHERAL_EVENT_INTERVAL_S = 1.0


def _load_config() -> dict:
    """Load calibration.json and return its contents, or defaults if missing."""
    defaults = {
        "osc_ip": _DEFAULT_OSC_IP,
        "osc_port": _DEFAULT_OSC_PORT,
        "osc_queue_fps": _DEFAULT_OSC_QUEUE_FPS,
        "osc_click_gap_ms": _DEFAULT_OSC_CLICK_GAP_MS,
        "projection_aspect": _DEFAULT_PROJECTION_ASPECT,
        "peripheral_margin": _DEFAULT_PERIPHERAL_MARGIN,
        "peripheral_event_interval_s": _DEFAULT_PERIPHERAL_EVENT_INTERVAL_S,
        "stream_fps": _DEFAULT_STREAM_FPS,
        "stream_width": _DEFAULT_STREAM_WIDTH,
        "stream_height": _DEFAULT_STREAM_HEIGHT,
        "jpeg_quality": _DEFAULT_JPEG_QUALITY,
    }
    if not os.path.exists(CALIB_FILE):
        return defaults
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Warning: {CALIB_FILE} is not a JSON object; using defaults")
            return defaults
        for key in defaults:
            if key in data:
                defaults[key] = data[key]
    except Exception as e:
        print(f"Warning: failed to load {CALIB_FILE}: {e}; using defaults")
    return defaults


_CONFIG = _load_config()
OSC_IP = _CONFIG["osc_ip"]
OSC_PORT = int(_CONFIG["osc_port"])
OSC_QUEUE_FPS = float(_CONFIG["osc_queue_fps"])
OSC_QUEUE_INTERVAL_S = 1.0 / OSC_QUEUE_FPS
OSC_CLICK_GAP_S = float(_CONFIG["osc_click_gap_ms"]) / 1000.0
PROJECTION_ASPECT = float(_CONFIG["projection_aspect"])
PERIPHERAL_MARGIN = float(_CONFIG["peripheral_margin"])
PERIPHERAL_EVENT_INTERVAL_S = float(_CONFIG["peripheral_event_interval_s"])
STREAM_FPS = float(_CONFIG["stream_fps"])
STREAM_INTERVAL_S = 1.0 / STREAM_FPS
STREAM_WIDTH = int(_CONFIG["stream_width"])
STREAM_HEIGHT = int(_CONFIG["stream_height"])
STREAM_JPEG_QUALITY = int(_CONFIG["jpeg_quality"])


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def save_calibration(src_points: np.ndarray, extra: dict | None = None) -> None:
    """Save the 4 source points (in capture-resolution pixels) to a file.

    Preserves any additional keys already present in calibration.json
    (e.g. OSC/network settings).
    """
    data: dict = {}
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception as e:
            print(f"Warning: could not read existing {CALIB_FILE}: {e}")
            data = {}
    if extra:
        data.update(extra)
    data["points"] = src_points.tolist()
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {CALIB_FILE}")


def load_calibration() -> np.ndarray | None:
    """Load previously saved calibration points, or return None."""
    if not os.path.exists(CALIB_FILE):
        return None
    try:
        with open(CALIB_FILE) as f:
            data = json.load(f)
        pts = np.array(data["points"], dtype=np.float32)
        if pts.shape == (4, 2):
            print(f"Calibration loaded from {CALIB_FILE}")
            return pts
    except Exception as e:
        print(f"Failed to load calibration: {e}")
    return None


def compute_homography(src_points: np.ndarray, cap_w: int, cap_h: int):
    """Compute homography mapping 4 src points to [-1,1] x [-1,1].

    TL->(-1,-1), TR->(1,-1), BR->(1,1), BL->(-1,1).
    Points outside the calibrated area will have coordinates beyond ±1.
    """
    dst = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    H, _ = cv2.findHomography(src_points, dst)
    return H, None, None, None, None, None


def apply_homography(H: np.ndarray, cx: float, cy: float):
    """Transform a point to [-1,1] normalised coordinates."""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    nx, ny = float(out[0][0][0]), float(out[0][0][1])
    return nx, ny


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    """Thread-safe parameter state shared between capture and network threads."""

    def __init__(self) -> None:
        self.threshold = INITIAL_THRESHOLD
        self.invert = True
        self.blur_kernel = BLUR_KERNEL_INITIAL
        self.min_area = MIN_AREA_INITIAL
        self.lock = threading.Lock()
        self.running = True
        # Homography matrix (None = no calibration)
        self.homography: np.ndarray | None = None
        # Raw src points in capture-resolution pixels
        self.calib_src: np.ndarray | None = None

    def set_calibration(self, src_points: np.ndarray) -> None:
        cap_w, cap_h = self.capture_size
        H, *_ = compute_homography(src_points, cap_w, cap_h)
        with self.lock:
            self.calib_src = src_points
            self.homography = H
        save_calibration(src_points)
        print("Homography updated")

    def get_homography(self) -> np.ndarray | None:
        with self.lock:
            return self.homography

    def adjust_threshold(self, delta: int) -> None:
        with self.lock:
            self.threshold = max(0, min(255, self.threshold + delta))

    def adjust_blur(self, delta: int) -> None:
        with self.lock:
            self.blur_kernel = max(
                BLUR_KERNEL_MIN,
                min(BLUR_KERNEL_MAX, self.blur_kernel + delta),
            )
            if self.blur_kernel % 2 == 0:
                self.blur_kernel += 1 if delta > 0 else -1

    def adjust_min_area(self, delta: int) -> None:
        with self.lock:
            self.min_area = max(0, self.min_area + delta)

    def toggle_invert(self) -> None:
        with self.lock:
            self.invert = not self.invert

    def get(self):
        with self.lock:
            return self.threshold, self.invert, self.blur_kernel, self.min_area


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def find_centroids(binary_image, min_area: int):
    """Return up to MAX_OBJECTS centroids sorted by descending area."""
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centroids.append((cx, cy, area))
    centroids.sort(key=lambda item: item[2], reverse=True)
    return centroids[:MAX_OBJECTS]


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

def handle_client(conn: socket.socket, state: SharedState) -> None:
    """Read commands from a remote client.

    Protocol:
      0x00 (or any printable byte) = single-char key command (1 byte total)
      0x01 = calibration frame: 1 + 4(length) + JSON
    """
    try:
        conn.settimeout(0.05)
        while state.running:
            try:
                header = conn.recv(1)
            except socket.timeout:
                continue
            if not header:
                break

            if header == b"\x01":
                # Calibration payload
                try:
                    size_data = _recv_exact(conn, 4)
                    length = struct.unpack("!I", size_data)[0]
                    payload = _recv_exact(conn, length)
                    data = json.loads(payload)
                    norm_pts = data.get("calib", [])
                    if len(norm_pts) == 4:
                        # norm_pts are relative to display panel; convert to
                        # capture resolution.
                        # We stored display_size on state for this purpose.
                        panel_w, panel_h = state.display_size
                        cap_w, cap_h = state.capture_size
                        src = np.array(
                            [
                                [p["x"] * cap_w, p["y"] * cap_h]
                                for p in norm_pts
                            ],
                            dtype=np.float32,
                        )
                        state.set_calibration(src)
                except Exception as e:
                    print(f"Calibration receive error: {e}")
            else:
                key = header[0]
                if key == ord("q"):
                    state.running = False
                elif key in (ord("w"), ord("+")):
                    state.adjust_threshold(THRESHOLD_STEP)
                elif key in (ord("s"), ord("-")):
                    state.adjust_threshold(-THRESHOLD_STEP)
                elif key == ord("a"):
                    state.adjust_blur(BLUR_KERNEL_STEP)
                elif key == ord("z"):
                    state.adjust_blur(-BLUR_KERNEL_STEP)
                elif key == ord("d"):
                    state.adjust_min_area(MIN_AREA_STEP)
                elif key == ord("c"):
                    state.adjust_min_area(-MIN_AREA_STEP)
                elif key == ord("i"):
                    state.toggle_invert()
                elif key == ord("p"):
                    with state.lock:
                        state.calib_src = None
                        state.homography = None
                    print("Calibration reset")
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise EOFError("Connection closed")
        data += chunk
    return data


def send_frame(conn: socket.socket, frame: bytes) -> bool:
    """Send a JPEG frame prefixed with its size."""
    try:
        conn.setblocking(True)
        conn.sendall(struct.pack("!I", len(frame)) + frame)
        return True
    except OSError:
        return False



def accept_clients(server: socket.socket, state: SharedState, client_ref: list):
    """Accept incoming client connections in a background thread."""
    server.settimeout(0.5)
    while state.running:
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        print(f"Client connected: {addr}")
        with client_ref[1]:
            if client_ref[0] is not None:
                try:
                    client_ref[0].close()
                except OSError:
                    pass
            client_ref[0] = conn
        threading.Thread(
            target=handle_client, args=(conn, state), daemon=True
        ).start()


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def _make_warp_H(calib_src, cap_size, display_size):
    """Compute the same homography as the client's compute_local_homography."""
    pw, ph = display_size
    cap_w, cap_h = cap_size
    sx, sy = pw / cap_w, ph / cap_h
    pts_d = np.array(calib_src, dtype=np.float32) * np.array([sx, sy])
    dst = np.array([[0,0],[pw,0],[pw,ph],[0,ph]], dtype=np.float32)
    H, _ = cv2.findHomography(pts_d, dst)
    scale = 0.8
    tx = pw * (1 - scale) / 2.0
    ty = ph * (1 - scale) / 2.0
    S = np.array([[scale, 0, tx],
                  [0, scale, ty],
                  [0, 0,     1 ]], dtype=np.float64)
    return (S @ H.astype(np.float64)).astype(np.float32)


def _warp_panel(panel: np.ndarray, H: np.ndarray, display_size) -> np.ndarray:
    pw, ph = display_size
    return cv2.warpPerspective(panel, H, (pw, ph))


def build_preview(
    frame,
    binary,
    centroids,
    display_size,
    threshold_value,
    blur_kernel,
    min_area,
    invert,
    calib_src,
    cap_size=None,
    H_warp=None,
):
    pw, ph = display_size
    frame_display = cv2.resize(frame, display_size)
    binary_display = cv2.resize(binary, display_size)
    binary_rgb = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2RGB)

    cap_w, cap_h = cap_size if cap_size is not None else (binary.shape[1], binary.shape[0])
    sx_d, sy_d = pw / cap_w, ph / cap_h

    # Warp first.
    if H_warp is not None:
        frame_display = _warp_panel(frame_display, H_warp, display_size)
        binary_rgb    = _warp_panel(binary_rgb,    H_warp, display_size)

    preview = cv2.hconcat([frame_display, binary_rgb])

    # All overlays drawn AFTER warp on the final image.

    # Calibration quad: transform calib points through H_warp.
    if calib_src is not None:
        pts_d = np.array(calib_src, dtype=np.float32) * np.array([sx_d, sy_d])
        if H_warp is not None:
            pts_w = cv2.perspectiveTransform(pts_d.reshape(-1, 1, 2), H_warp).astype(np.int32)
        else:
            pts_w = pts_d.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(preview, [pts_w], isClosed=True, color=(0, 255, 255), thickness=2)
        pts_w_r = pts_w.copy(); pts_w_r[:, :, 0] += pw
        cv2.polylines(preview, [pts_w_r], isClosed=True, color=(0, 255, 255), thickness=2)

    # Centroids: transform through H_warp.
    for cx, cy, _ in centroids:
        pt = np.array([[[cx * sx_d, cy * sy_d]]], dtype=np.float32)
        if H_warp is not None:
            out = cv2.perspectiveTransform(pt, H_warp)
            px, py = int(out[0, 0, 0]), int(out[0, 0, 1])
        else:
            px, py = int(cx * sx_d), int(cy * sy_d)
        cv2.circle(preview, (px,      py), 5, (0, 0, 255), -1)
        cv2.circle(preview, (px + pw, py), 5, (0, 0, 255), -1)

    # Status text.
    calib_label = "CAL" if calib_src is not None else "---"
    flip_label  = "FLIP" if invert else "NRM"
    text = f"thr:{threshold_value} blur:{blur_kernel} min:{min_area} [{flip_label}][{calib_label}]  w/s:thr a/z:blur d/c:min i:flip q:quit"
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    show_gui = "--gui" in sys.argv
    state = SharedState()

    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"OSC target: {OSC_IP}:{OSC_PORT} (aspect={PROJECTION_ASPECT}, "
          f"queue_fps={OSC_QUEUE_FPS}, click_gap_ms={OSC_CLICK_GAP_S * 1000}, "
          f"peripheral_margin={PERIPHERAL_MARGIN}, "
          f"peripheral_interval={PERIPHERAL_EVENT_INTERVAL_S}s)")

    # Track the last time a peripheral event was sent per touch id.
    peripheral_last_sent: dict[int, float] = {}

    # OSC outbound queue: flushed at OSC_QUEUE_FPS.
    # Each touch appends a (1, tx, ty) then (0, tx, ty) pair.
    osc_queue: list[tuple[int, float, float]] = []
    osc_last_flush = 0.0
    osc_was_active = False

    camera = Picamera2(camera_num=0)
    sensor_modes = camera.sensor_modes
    max_size = max(
        (mode["size"] for mode in sensor_modes),
        key=lambda s: s[0] * s[1],
    )
    config = camera.create_video_configuration(
        main={"size": max_size, "format": "RGB888"},
        controls={
            "AeEnable": False,
            "ExposureTime": EXPOSURE_TIME_US,
            "AnalogueGain": ANALOG_GAIN,
            "FrameDurationLimits": (FRAME_DURATION_US, FRAME_DURATION_US),
        },
    )
    camera.configure(config)
    camera.start()

    half_width = STREAM_WIDTH // 2
    display_size = (half_width, STREAM_HEIGHT)

    # Store sizes on state so the client handler can convert coordinates.
    state.display_size = display_size
    state.capture_size = max_size  # (width, height)

    # Load saved calibration if available.
    saved = load_calibration()
    if saved is not None:
        state.set_calibration(saved)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((SERVER_HOST, SERVER_PORT))
    server.listen(1)
    print(f"Server listening on port {SERVER_PORT}")

    client_ref = [None, threading.Lock()]
    accept_thread = threading.Thread(
        target=accept_clients, args=(server, state, client_ref), daemon=True
    )
    accept_thread.start()

    try:
        H_warp_cached = None
        calib_src_cached = None
        stream_last_sent = 0.0

        while state.running:
            frame = camera.capture_array()
            threshold_value, invert, blur_kernel, min_area = state.get()

            if invert:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            grayscale = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(grayscale, (blur_kernel, blur_kernel), 0)
            _, binary = cv2.threshold(
                blurred, threshold_value, 255, cv2.THRESH_BINARY
            )

            centroids = find_centroids(binary, min_area)

            H = state.get_homography()
            calib_src = state.calib_src

            # Recompute display warp matrix only when calibration changes.
            if calib_src is not calib_src_cached:
                calib_src_cached = calib_src
                H_warp_cached = (
                    _make_warp_H(calib_src, max_size, display_size)
                    if calib_src is not None else None
                )

            now = time.monotonic()
            if centroids:
                objects = []
                for idx, (cx, cy, area) in enumerate(centroids, start=1):
                    obj = {"x": int(cx), "y": int(cy), "area": float(area)}
                    if H is not None:
                        nx, ny = apply_homography(H, cx, cy)
                        # TouchDesigner coordinates: x scaled by aspect, y flipped
                        # so +1 is the top of the screen and -1 is the bottom.
                        tx = round(max(-1.0, min(1.0, nx)) * PROJECTION_ASPECT, 4)
                        ty = round(max(-1.0, min(1.0, -ny)), 4)
                        obj["nx"] = round(nx, 4)
                        obj["ny"] = round(ny, 4)

                        # Discard touches that fall outside the calibrated
                        # projection area (±1.1 in normalised space).
                        if abs(nx) > 1.1 or abs(ny) > 1.1:
                            obj["osc_discarded"] = True
                            objects.append(obj)
                            continue

                        # Determine whether the touch is in the peripheral
                        # margin.  Points beyond ±1 are also considered
                        # peripheral because they lie outside the calibrated
                        # area.
                        abs_nx = abs(nx)
                        abs_ny = abs(ny)
                        peripheral = (
                            abs_nx >= (1.0 - PERIPHERAL_MARGIN)
                            or abs_ny >= (1.0 - PERIPHERAL_MARGIN)
                        )

                        if peripheral:
                            last = peripheral_last_sent.get(idx, 0.0)
                            if now - last < PERIPHERAL_EVENT_INTERVAL_S:
                                # Throttle: skip this peripheral event.
                                obj["osc_throttled"] = True
                                objects.append(obj)
                                continue
                            peripheral_last_sent[idx] = now

                        print(f"[OSC] queue /touch tx={tx} ty={ty} (nx={nx}, ny={ny})")
                        osc_queue.append((1, tx, ty))
                        osc_queue.append((0, tx, ty))
                        osc_was_active = True
                    objects.append(obj)
                payload = {
                    "centroids": objects,
                    "threshold": threshold_value,
                    "blur": blur_kernel,
                    "min_area": min_area,
                    "calibrated": H is not None,
                }
                print(json.dumps(payload, ensure_ascii=False))
                sys.stdout.flush()

            # Flush queued OSC messages at the configured rate.
            # Each touch is a (1,x,y)/(0,x,y) pair; send 1, wait, send 0.
            if now - osc_last_flush >= OSC_QUEUE_INTERVAL_S:
                osc_last_flush = now
                if osc_queue:
                    it = iter(osc_queue)
                    for msg_on, msg_off in zip(it, it):
                        _, tx_on, ty_on = msg_on
                        _, tx_off, ty_off = msg_off
                        osc_client.send_message("/touch", [1, tx_on, ty_on])
                        time.sleep(OSC_CLICK_GAP_S)
                        osc_client.send_message("/touch", [0, tx_off, ty_off])
                    print(f"[OSC] flushed {len(osc_queue) // 2} touch(es)")
                    osc_queue.clear()
                    osc_was_active = True
                elif osc_was_active:
                    osc_client.send_message("/touch", [0, 0.0, 0.0])
                    print("[OSC] idle: sent active=0")
                    osc_was_active = False

            if now - stream_last_sent >= STREAM_INTERVAL_S:
                stream_last_sent = now
                preview = build_preview(
                    frame, binary, centroids, display_size,
                    threshold_value, blur_kernel, min_area, invert, calib_src,
                    cap_size=max_size, H_warp=H_warp_cached,
                )

                ok, encoded = cv2.imencode(
                    ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
                )
                if ok:
                    jpeg_bytes = encoded.tobytes()
                    with client_ref[1]:
                        conn = client_ref[0]
                    if conn is not None:
                        if not send_frame(conn, jpeg_bytes):
                            with client_ref[1]:
                                client_ref[0] = None

                if show_gui:
                    cv2.imshow("NoIR Camera: original | binary", preview)

            if show_gui:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0xFF

            if key == ord("q"):
                state.running = False
            elif key in (ord("w"), ord("+")):
                state.adjust_threshold(THRESHOLD_STEP)
            elif key in (ord("s"), ord("-")):
                state.adjust_threshold(-THRESHOLD_STEP)
            elif key == ord("a"):
                state.adjust_blur(BLUR_KERNEL_STEP)
            elif key == ord("z"):
                state.adjust_blur(-BLUR_KERNEL_STEP)
            elif key == ord("d"):
                state.adjust_min_area(MIN_AREA_STEP)
            elif key == ord("c"):
                state.adjust_min_area(-MIN_AREA_STEP)
            elif key == ord("i"):
                state.toggle_invert()

            time.sleep(0.001)
    finally:
        state.running = False
        with client_ref[1]:
            if client_ref[0] is not None:
                try:
                    client_ref[0].close()
                except OSError:
                    pass
        try:
            server.close()
        except OSError:
            pass
        camera.stop()
        if show_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
