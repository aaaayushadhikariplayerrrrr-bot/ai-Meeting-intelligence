# File: C:/Users/gurpr/Downloads/Multilingual Meeting Intelligence System/app/__init__.py
import os
import sys
from pathlib import Path

# Resolve the absolute path to the real backend application code.
# __file__ → …/app/__init__.py
# One level up → project root (…/)
# Then into backend/app
package_dir = (Path(__file__).parent.parent / "backend" / "app").resolve()

if not package_dir.is_dir():
    raise RuntimeError(f"Expected backend/app directory not found at {package_dir}")

# Prepend the path so it takes precedence over any other “app” modules.
sys.path.insert(0, str(package_dir))

# Expose the package as a namespace package (optional but helpful for type‑checkers)
__path__ = [str(package_dir)]