"""Pytest configuration to ensure the package in ``src/`` is importable.

Now that tests live in a top-level ``tests/`` folder (outside ``src``), we need
to ensure ``src`` is on ``sys.path`` when invoking ``pytest`` without an
editable install.
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
