"""Put video/ on sys.path so tests import its top-level modules (video/ is not a package)."""

import sys
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[1]
if str(VIDEO_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEO_DIR))
