"""Multilingual Meeting Intelligence System - Backend Application Package."""
import os

# Phase 9.3.1: Establish CPU thread environment defaults at earliest package entry point,
# before any native numerical libraries (OpenMP, OpenBLAS, MKL, PyTorch, CTranslate2) load.
# Preserves existing user environment overrides. Does NOT import PyTorch or heavy libraries.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _var not in os.environ:
        os.environ[_var] = os.environ.get("CPU_MAX_THREADS", "4")

__version__ = "0.1.0"
