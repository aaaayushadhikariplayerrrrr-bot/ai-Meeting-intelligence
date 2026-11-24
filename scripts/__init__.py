# Package shim to make backend/scripts importable as top-level 'scripts'
import sys
from pathlib import Path

# Determine the real backend/scripts directory relative to this file
backend_scripts_dir = Path(__file__).resolve().parent / "backend" / "scripts"
if not backend_scripts_dir.is_dir():
    # If the structure is different, try one level higher
    backend_scripts_dir = Path(__file__).resolve().parent.parent / "backend" / "scripts"

# Ensure the directory is on sys.path (helps absolute imports)
if str(backend_scripts_dir) not in sys.path:
    sys.path.insert(0, str(backend_scripts_dir))

# Extend this package's search path so submodule imports resolve correctly
__path__ = [str(backend_scripts_dir)]
