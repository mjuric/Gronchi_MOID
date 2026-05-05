"""Make the in-tree package importable when the user hasn't pip-installed it.

When the package is properly installed (`pip install .`), this is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent / "python"
if PKG_DIR.exists() and str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
