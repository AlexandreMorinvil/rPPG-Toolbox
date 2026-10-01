"""Start the launcher: python -m launcher [--port 8790] (run from the rPPG-Toolbox folder)."""

import argparse
import sys
import webbrowser

from .server import LauncherServer


def main():
    parser = argparse.ArgumentParser(description="rPPG-Toolbox experiment launcher (web UI).")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8790, help="Port (default: 8790).")
    parser.add_argument("--allow-host", action="append", default=[],
                        help="Extra host name allowed in the Host header (e.g. a machine name), repeatable.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    args = parser.parse_args()

    allowed = {f"{name}:{args.port}" for name in ("127.0.0.1", "localhost", "[::1]", args.host, *args.allow_host)}
    try:
        server = LauncherServer((args.host, args.port), allowed)
    except OSError as error:
        sys.exit(f"Cannot listen on {args.host}:{args.port}: {error}. Choose another --port.")
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}/"
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: the launcher can start processes on this machine. Only expose it on trusted networks.")
    print(f"rPPG-Toolbox launcher running at {url}")
    print("Jobs keep running if you close the launcher. Press Ctrl+C to stop the web server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
