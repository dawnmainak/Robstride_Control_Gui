#!/usr/bin/env python3
"""Launcher for the RobStride control GUI.

Run with:  python3 main.py
"""

from __future__ import annotations

import logging
import os
import sys


def _configure_logging() -> None:
    """Enable DEBUG logging (incl. poll_status TX/RX frames) when requested.

    Off by default. Set ROBSTRIDE_DEBUG=1 to see per-poll serial frames:
        ROBSTRIDE_DEBUG=1 python3 main.py
    """
    debug = os.environ.get("ROBSTRIDE_DEBUG", "").lower() in ("1", "true", "yes")
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    _configure_logging()

    import argparse
    ap = argparse.ArgumentParser(description="RobStride control GUI")
    ap.add_argument("--remote", metavar="USER@HOST",
                    help="drive motors on a remote Pi over SSH instead of a "
                         "locally attached adapter, e.g. usvz@rs04pi.local")
    ap.add_argument("--remote-path", default="~/Desktop/Robstride_Control_Gui/rpi_agent.py",
                    help="path to rpi_agent.py on the remote machine")
    args = ap.parse_args()
    remote = (args.remote, args.remote_path) if args.remote else None
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(
            "PySide6 is not installed.\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
        )
        return 1

    from robstride_gui.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("RobStride Control")
    window = MainWindow(remote=remote)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())