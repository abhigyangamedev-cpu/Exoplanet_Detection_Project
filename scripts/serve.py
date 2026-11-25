"""Start the model-checker website on this machine and open it in the browser.

Pick any labelled KOI and see what each model said about it and whether that was right. The
site only reads saved scores (out-of-fold for development KOIs, the held-out score for test
KOIs); it never runs a model. Needs the ``web`` extra: ``pip install -e ".[web]"``.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser

from exoplanet.config import load_config
from exoplanet.web.app import create_app

HOST = "127.0.0.1"


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((HOST, port)) != 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    if not port_is_free(args.port):
        sys.exit(f"Port {args.port} is already in use; pick another with --port.")

    app = create_app(load_config(args.config))
    url = f"http://{HOST}:{args.port}"
    print(f"Model checker: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    app.run(host=HOST, port=args.port, debug=False)


if __name__ == "__main__":
    main()
