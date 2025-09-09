"""Pytest configuration to ensure the package in src/ is importable.

Without installing the project (editable or wheel), running `pytest` from the
repository root won't put the `src/` directory on `sys.path`, so
`import easy_local_features` fails with ModuleNotFoundError.

This file prepends the absolute path to `src/` so tests can import the package
when executed via `pytest` directly (e.g. `uv tool run pytest`).
"""

from __future__ import annotations
import sys
from pathlib import Path

# Path to this file: <repo>/src/test/conftest.py
# We want to add <repo>/src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1]  # parents[1] == <repo>/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
