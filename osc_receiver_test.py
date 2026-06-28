#!/usr/bin/env python3
"""Simple OSC receiver to verify /touch messages from camera_server.py.

Run this on the TouchDesigner PC or any machine reachable from the Pi:

    python3 osc_receiver_test.py

Then start camera_server.py with OSC_IP pointing to this machine.
"""

import argparse
import sys

from pythonosc import dispatcher, osc_server


def on_touch(addr, idx, x, y):
    print(f"/{addr} id={idx} x={x:.4f} y={y:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OSC /touch receiver test")
    parser.add_argument("--ip", default="0.0.0.0", help="IP to listen on")
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on")
    args = parser.parse_args()

    disp = dispatcher.Dispatcher()
    disp.map("/touch", on_touch)

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)
    print(f"Listening for OSC on {server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()
