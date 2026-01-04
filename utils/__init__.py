"""
Compatibility wrapper to expose the `protclassify` package via the legacy
`utils` namespace used in older notebooks.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
