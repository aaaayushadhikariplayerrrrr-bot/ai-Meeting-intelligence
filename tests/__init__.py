# Top-level package to expose backend/tests as importable 'tests'
import sys
from pathlib import Path

# Resolve path to the actual tests directory inside backend
backend_tests_dir = Path(__file__).resolve().parent / "backend" / "tests"
if not backend_tests_dir.is_dir():
    # Fallback in case relative path differs; try two levels up
    backend_tests_dir = Path(__file__).resolve().parent.parent / "backend" / "tests"

# Extend package __path__ so that submodule imports search the backend/tests directory
if str(backend_tests_dir) not in __path__:
    __path__.append(str(backend_tests_dir))
