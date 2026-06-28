#!/usr/bin/env python3
"""Remote client for camera_server.py.

Displays the streamed preview and forwards key presses to the server.

Key commands:
  w/s or +/- : increase/decrease threshold
  a/z        : increase/decrease Gaussian blur kernel
  d/c        : increase/decrease minimum object area
  i          : toggle 180-degree rotation
  p          : enter / restart calibration mode (click 4 points: top-left,
               top-right, bottom-right, bottom-left on the LEFT panel)
  q          : quit
"""

import json
import socket
import struct
import sys
import threading

import cv2
import numpy as np


HOST = "192.168.1.1"  # Change to the server's IP address.
PORT = 12345

CALIB_LABELS = ["TL", "TR", "BR", "BL"]
CALIB_COLORS = [
    (0, 255, 255),   # TL: yellow
    (0, 165, 255),   # TR: orange
    (0, 0, 255),     # BR: red
    (255, 0, 255),   # BL: magenta
]

# Destination for perspective warp: fills the left panel exactly.
DST_MARGIN = 0  # pixels of margin inside the panel


def receive_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("Connection closed")
        data += chunk
    return data


def send_key(conn: socket.socket, key: int) -> None:
    try:
        conn.sendall(bytes([key]))
    except OSError:
        pass


def send_calibration(conn: socket.socket, points: list, panel_size: tuple) -> None:
    """Send calibration points normalised to the left panel dimensions."""
    pw, ph = panel_size
    norm = [{"x": x / pw, "y": y / ph} for x, y in points]
    payload = json.dumps({"calib": norm}).encode()
    try:
        conn.sendall(b"\x01" + struct.pack("!I", len(payload)) + payload)
    except OSError:
        pass


def compute_local_homography(points: list, panel_w: int, panel_h: int):
    """Map 4 clicked points -> panel corners, then scale down to 80% centered."""
    src = np.array(points, dtype=np.float32)
    dst = np.array([
        [0,       0      ],
        [panel_w, 0      ],
        [panel_w, panel_h],
        [0,       panel_h],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)

    scale = 0.8
    tx = panel_w * (1 - scale) / 2.0
    ty = panel_h * (1 - scale) / 2.0
    S = np.array([[scale, 0,     tx],
                  [0,     scale, ty],
                  [0,     0,     1 ]], dtype=np.float64)
    H2 = (S @ H.astype(np.float64)).astype(np.float32)

    return H2, panel_w, panel_h, None, None, None


class CalibState:
    def __init__(self):
        self.active = False
        self.points = []
        self.panel_width = 0
        self.panel_height = 480
        self.latest_frame = None
        self.homography = None
        self.canvas_w = 0
        self.canvas_h = 0
        self.lock = threading.Lock()


def mouse_callback(event, x, y, flags, calib: CalibState):
    if not calib.active:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        with calib.lock:
            if calib.panel_width > 0 and len(calib.points) < 4:
                # Normalise click to left-panel coordinates regardless of which panel was clicked.
                lx = x if x < calib.panel_width else x - calib.panel_width
                calib.points.append((lx, y))
                if calib.latest_frame is not None:
                    display = calib.latest_frame.copy()
                    _draw_calibration(display, calib)
                    cv2.imshow("Remote Camera", display)


def _draw_calibration(frame, calib: CalibState) -> None:
    n = len(calib.points)
    pw = calib.panel_width
    for i, (px, py) in enumerate(calib.points):
        # Draw on left panel.
        cv2.circle(frame, (px, py), 6, CALIB_COLORS[i], -1)
        cv2.putText(frame, CALIB_LABELS[i], (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CALIB_COLORS[i], 2, cv2.LINE_AA)
        # Draw on right panel.
        cv2.circle(frame, (px + pw, py), 6, CALIB_COLORS[i], -1)
        cv2.putText(frame, CALIB_LABELS[i], (px + pw + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, CALIB_COLORS[i], 2, cv2.LINE_AA)
    if n < 4:
        msg = f"CALIB: click {CALIB_LABELS[n]} (either panel)  p=restart  ESC=cancel"
    else:
        msg = "CALIB: 4 points set. ENTER=confirm  p=restart  ESC=cancel"
    cv2.putText(
        frame, msg, (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )


def apply_warp(frame, calib: CalibState) -> np.ndarray:
    """Warp to canvas size then resize to panel so all source pixels are visible."""
    pw = calib.panel_width
    ph = calib.panel_height
    cw = calib.canvas_w
    ch = calib.canvas_h
    H = calib.homography
    if cw <= 0 or ch <= 0 or H is None:
        return frame.copy()
    left  = frame[:, :pw]
    right = frame[:, pw:]
    warped_left  = cv2.warpPerspective(left,  H, (cw, ch))
    warped_right = cv2.warpPerspective(right, H, (cw, ch))
    # Resize canvas to panel with letterbox (maintains aspect ratio).
    scale = min(pw / cw, ph / ch)
    ow, oh = int(cw * scale), int(ch * scale)
    resized_left  = cv2.resize(warped_left,  (ow, oh))
    resized_right = cv2.resize(warped_right, (ow, oh))
    # Pad to panel size with black borders.
    out_left  = np.zeros((ph, pw, 3), dtype=np.uint8)
    out_right = np.zeros((ph, pw, 3), dtype=np.uint8)
    x0 = (pw - ow) // 2
    y0 = (ph - oh) // 2
    out_left [y0:y0+oh, x0:x0+ow] = resized_left
    out_right[y0:y0+oh, x0:x0+ow] = resized_right
    return cv2.hconcat([out_left, out_right])


# ---------------------------------------------------------------------------
# Shared frame buffer
# ---------------------------------------------------------------------------

class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.updated = False
        self.lock = threading.Lock()
        self.running = True

    def put(self, frame):
        with self.lock:
            self.frame = frame
            self.updated = True

    def get(self):
        with self.lock:
            f = self.frame
            self.updated = False
            return f

    def has_update(self):
        with self.lock:
            return self.updated


def receive_loop(conn: socket.socket, buf: FrameBuffer) -> None:
    try:
        while buf.running:
            size_data = receive_exact(conn, 4)
            img_size = struct.unpack("!I", size_data)[0]
            img_data = receive_exact(conn, img_size)
            frame = cv2.imdecode(
                np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame is not None:
                buf.put(frame)
    except Exception:
        buf.running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    host = sys.argv[1] if len(sys.argv) > 1 else HOST

    # Show connection status window while trying to connect.
    cv2.namedWindow("Remote Camera")
    cv2.imshow("Remote Camera", np.zeros((480, 1280, 3), dtype=np.uint8))
    cv2.waitKey(1)

    RETRY_INTERVAL = 2.0   # seconds between retries
    conn = None
    while conn is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((host, PORT))
            s.settimeout(None)
            conn = s
            print(f"Connected to {host}:{PORT}")
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            s.close()
            msg = f"Connecting to {host}:{PORT} ...  ({type(e).__name__})"
            print(msg)
            img = np.zeros((480, 1280, 3), dtype=np.uint8)
            cv2.putText(img, msg, (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "Press Q to quit", (30, 290),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.imshow("Remote Camera", img)
            key = cv2.waitKey(int(RETRY_INTERVAL * 1000)) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return

    calib = CalibState()
    buf = FrameBuffer()

    cv2.setMouseCallback("Remote Camera", mouse_callback, calib)
    cv2.imshow("Remote Camera", np.zeros((480, 1280, 3), dtype=np.uint8))
    cv2.waitKey(1)

    recv_thread = threading.Thread(target=receive_loop, args=(conn, buf), daemon=True)
    recv_thread.start()

    def enter_calib_mode():
        with calib.lock:
            calib.active = True
            calib.points = []
            calib.homography = None  # reset warp so raw frame is shown
            calib.canvas_w = 0
            calib.canvas_h = 0
        print("Calibration mode: click TL, TR, BR, BL on the left panel")
        with calib.lock:
            lf = calib.latest_frame
        if lf is not None:
            display = lf.copy()
            _draw_calibration(display, calib)
            cv2.imshow("Remote Camera", display)

    try:
        while buf.running:
            if buf.has_update():
                frame = buf.get()
                calib.panel_width = frame.shape[1] // 2
                calib.panel_height = frame.shape[0]
                with calib.lock:
                    calib.latest_frame = frame.copy()
                    active = calib.active
                    H = calib.homography

                if active:
                    display = frame.copy()
                    _draw_calibration(display, calib)
                else:
                    display = frame.copy()
                cv2.imshow("Remote Camera", display)

            key = cv2.waitKey(16) & 0xFF

            if key == ord("p"):
                # p resets calibration on server and enters local calib mode
                send_key(conn, key)
                enter_calib_mode()
                continue

            if calib.active:
                if key == 27:  # ESC
                    with calib.lock:
                        calib.active = False
                        calib.points = []
                    print("Calibration cancelled")
                elif key == 13:  # ENTER
                    with calib.lock:
                        pts = list(calib.points)
                        pw = calib.panel_width
                        ph = calib.panel_height
                    if len(pts) == 4:
                        H, cw, ch, *_ = compute_local_homography(pts, pw, ph)
                        with calib.lock:
                            calib.homography = H
                            calib.canvas_w = cw
                            calib.canvas_h = ch
                            calib.active = False
                            calib.points = []
                        send_calibration(conn, pts, (pw, ph))
                        print(f"Calibration confirmed and sent: {pts}")
                continue

            if key == ord("q"):
                send_key(conn, key)
                break
            elif key in (
                ord("w"), ord("s"), ord("+"), ord("-"),
                ord("a"), ord("z"), ord("d"), ord("c"), ord("i"),
            ):
                send_key(conn, key)
    finally:
        buf.running = False
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
