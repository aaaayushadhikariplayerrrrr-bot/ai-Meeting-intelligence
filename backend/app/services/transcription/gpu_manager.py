"""GPU and Hardware Resource Manager for Speech Recognition.

Handles:
- Dynamic discovery and registration of CUDA & cuDNN DLLs on Windows
- Verification of CTranslate2 CUDA compute compatibility
- GPU VRAM monitoring and safety checks
- Fallback resolution when GPU is saturated or incompatible
"""

import os
import sys
import gc
import functools
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


def set_cpu_thread_environment_defaults(max_threads: Optional[int] = None) -> None:
    """Establish CPU thread environment defaults before any native numerical libraries load.

    Must be executed before importing torch, numpy, scipy, or ctranslate2.
    Preserves any existing environment overrides set by the user or host system.
    Does NOT import PyTorch or any heavy dependencies.
    """
    effective_max = str(
        max_threads
        or os.environ.get("CPU_MAX_THREADS")
        or "4"
    )
    for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if env_var not in os.environ:
            os.environ[env_var] = effective_max


# Establish environment defaults immediately before any native DLL / PyTorch discovery
set_cpu_thread_environment_defaults()


# Register DLL directories immediately on module import
def configure_windows_dll_directories() -> None:
    """Ensure CUDA and cuDNN runtime DLLs from PyTorch and CUDA Toolkit are discoverable on Windows."""
    if os.name != "nt":
        return

    # 1. PyTorch lib directory (contains cublas64_12.dll, cudnn64_9.dll, etc.)
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
                logger.debug(f"Added torch lib to DLL directory: {torch_lib}")
            except Exception as e:
                logger.warning(f"Failed to add torch lib to DLL directory: {e}")
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass

    # 2. CUDA Toolkit bin directory
    cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
    cuda_bin = os.path.join(cuda_path, "bin")
    if os.path.isdir(cuda_bin):
        try:
            os.add_dll_directory(cuda_bin)
        except Exception:
            pass
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")


def apply_pyav_compatibility_shim() -> None:
    """Ensure PyAV >= 14 does not raise TypeError with faster-whisper on metadata_errors argument.
    
    Safety constraints:
    - Only intercepts 'metadata_errors' keyword argument removed in PyAV >= 14.
    - Preserves function signature, docstring, and all positional/keyword arguments.
    - Does not alter audio decoding behavior, container opening, or format parsing.
    - Idempotent and thread-safe wrapper.
    """
    try:
        import av
        if hasattr(av, "open"):
            _orig_open = av.open
            if not getattr(_orig_open, "_is_compat_shim", False):
                @functools.wraps(_orig_open)
                def _compat_av_open(*args, **kwargs):
                    kwargs.pop("metadata_errors", None)
                    return _orig_open(*args, **kwargs)
                _compat_av_open._is_compat_shim = True
                _compat_av_open._orig_func = _orig_open
                av.open = _compat_av_open
                logger.debug("Applied PyAV metadata_errors compatibility shim.")
    except ImportError:
        pass


# Execute DLL setup and PyAV compatibility
configure_windows_dll_directories()
apply_pyav_compatibility_shim()


# --- CPU Resource Allocation & Thread Bounding (Phase 9.3) ---
_cpu_resources_configured: bool = False
_active_gpu_consumers: Dict[str, str] = {}


def configure_cpu_resources(
    intra_op_threads: Optional[int] = None,
    inter_op_threads: Optional[int] = None,
    max_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """Configure OpenMP, BLAS, and PyTorch CPU thread bounds safely and strictly once.

    CRITICAL INITIALIZATION ORDER & DOCUMENTATION:
    1. OpenMP / OpenBLAS / MKL threads (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS):
       Must be set in os.environ before the respective native C/C++ libraries initialize their thread pools.
       Once OpenMP or OpenBLAS creates its worker threads upon the first matrix multiply or model call,
       changing os.environ has no runtime effect.
    2. PyTorch inter-op threads (torch.set_num_interop_threads):
       Must be called strictly before any parallel PyTorch operations or thread pools start.
       Attempting to call it after parallel work starts raises:
       RuntimeError: Error: cannot set number of interop threads after parallel work has started
    3. PyTorch intra-op threads (torch.set_num_threads):
       Controls GEMM/tensor intra-operation parallelism. Bounding to a modest core count (e.g. 4)
       prevents PyTorch from monopolizing all CPU hardware threads and starving web requests or SQLite.
    4. Idempotency Guard:
       Prevents repeatedly thrashing thread pools during runtime or concurrent requests.
    """
    global _cpu_resources_configured
    if _cpu_resources_configured:
        return get_cpu_resource_info()

    # 1. OpenMP and BLAS environment variable defaults
    from app.core.config import settings
    effective_max = max_threads or getattr(settings, "CPU_MAX_THREADS", 4)
    set_cpu_thread_environment_defaults(max_threads=effective_max)

    # 2. PyTorch intra-op and inter-op threads
    try:
        import torch
        effective_intra = intra_op_threads or getattr(settings, "TORCH_NUM_THREADS", 4)
        effective_inter = inter_op_threads or getattr(settings, "TORCH_NUM_INTEROP_THREADS", 2)

        if effective_intra and effective_intra > 0:
            torch.set_num_threads(effective_intra)

        if effective_inter and effective_inter > 0:
            try:
                torch.set_num_interop_threads(effective_inter)
            except RuntimeError as e:
                logger.debug(f"PyTorch inter-op threads could not be modified (already active): {e}")
    except ImportError:
        pass

    _cpu_resources_configured = True
    logger.info(f"Initialized CPU resources: {get_cpu_resource_info()}")
    return get_cpu_resource_info()


def get_cpu_resource_info() -> Dict[str, Any]:
    """Inspect current CPU and threading configuration across PyTorch, BLAS, and environment."""
    info: Dict[str, Any] = {
        "cpu_count": os.cpu_count() or 1,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_intra_op_threads": None,
        "torch_inter_op_threads": None,
        "active_gpu_consumers": dict(_active_gpu_consumers),
    }
    try:
        import torch
        info["torch_intra_op_threads"] = torch.get_num_threads()
        info["torch_inter_op_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return info


def register_active_gpu_consumer(consumer_id: str, model_name: str) -> None:
    """Register an active model consumer using GPU VRAM."""
    _active_gpu_consumers[consumer_id] = model_name


def unregister_active_gpu_consumer(consumer_id: str) -> None:
    """Unregister a model consumer when unloaded from GPU."""
    _active_gpu_consumers.pop(consumer_id, None)


def can_use_gpu_for_embedding(min_free_vram_mb: float = 2000.0) -> Tuple[bool, str]:
    """Assess whether GPU VRAM is available for embedding models without competing with ASR on 8 GB GPU."""
    # Never co-locate E5 on GPU while Faster-Whisper is active on 8 GB GPU
    if "faster-whisper" in _active_gpu_consumers:
        return (
            False,
            f"GPU is allocated to Faster-Whisper ({_active_gpu_consumers.get('faster-whisper')}); "
            "embedding remains on CPU to avoid GPU VRAM contention on 8 GB GPU."
        )

    gpu_info = check_gpu_hardware()
    if not gpu_info.get("torch_cuda_available", False):
        return False, "CUDA is not available on this host."

    free_vram = gpu_info.get("free_vram_mb", 0.0)
    if free_vram < min_free_vram_mb:
        return False, f"Insufficient free VRAM ({free_vram:.1f} MB free, requires {min_free_vram_mb:.1f} MB)."

    return True, f"GPU permitted for embedding ({free_vram:.1f} MB free VRAM)."


# Execute initial CPU thread bounding
configure_cpu_resources()



def check_gpu_hardware() -> Dict[str, Any]:
    """Inspect NVIDIA GPU device and PyTorch CUDA environment."""
    info: Dict[str, Any] = {
        "torch_cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "compute_capability": None,
        "total_vram_mb": 0.0,
        "free_vram_mb": 0.0,
        "allocated_vram_mb": 0.0,
        "cudnn_available": False,
        "cudnn_version": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["torch_cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = torch.cuda.get_device_capability(0)

            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            info["total_vram_mb"] = round(total, 2)
            info["allocated_vram_mb"] = round(allocated, 2)
            info["free_vram_mb"] = round(total - reserved, 2)

            info["cudnn_available"] = torch.backends.cudnn.is_available()
            if info["cudnn_available"]:
                info["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        pass

    return info


def check_ctranslate2_support() -> Tuple[bool, str, List[str]]:
    """Check if CTranslate2 can natively perform CUDA operations.
    
    Returns:
        (is_supported, diagnostic_message, list_of_supported_compute_types)
    """
    configure_windows_dll_directories()
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        if cuda_count == 0:
            return (
                False,
                "CTranslate2 reports 0 CUDA devices available.",
                list(ctranslate2.get_supported_compute_types("cpu"))
            )

        supported_cuda_types = list(ctranslate2.get_supported_compute_types("cuda"))
        return (
            True,
            f"CTranslate2 detects {cuda_count} CUDA device(s). Supported compute types: {supported_cuda_types}",
            supported_cuda_types
        )
    except ImportError:
        return False, "CTranslate2 is not installed.", []
    except Exception as e:
        return False, f"CTranslate2 CUDA inspection error: {str(e)}", []


def select_optimal_device_and_compute(
    preferred_device: str = "cuda",
    preferred_compute: str = "float16",
    min_free_vram_mb: float = 2000.0
) -> Tuple[str, str, str]:
    """Determine the actual device and compute type to use based on hardware reality.
    
    Returns:
        (selected_device, selected_compute_type, reasoning_note)
    """
    if preferred_device.lower() == "cpu":
        return "cpu", "int8", "CPU explicitly requested by configuration"

    # Check CTranslate2 CUDA support
    is_cuda_ready, reason, supported_types = check_ctranslate2_support()
    if not is_cuda_ready:
        logger.warning(f"CUDA requested but CTranslate2 cannot use GPU: {reason}. Falling back to CPU.")
        return "cpu", "int8", f"Fallback to CPU: {reason}"

    # Check VRAM headroom
    gpu_info = check_gpu_hardware()
    free_vram = gpu_info.get("free_vram_mb", 0.0)
    if free_vram < min_free_vram_mb:
        logger.warning(
            f"Low GPU VRAM: {free_vram:.1f} MB available, requires {min_free_vram_mb} MB. Falling back to CPU."
        )
        return "cpu", "int8", f"Fallback to CPU: Low VRAM ({free_vram:.1f} MB free)"

    # Validate compute type
    compute = preferred_compute
    if compute not in supported_types:
        # Pick best available
        for candidate in ["float16", "int8_float16", "bfloat16", "int8"]:
            if candidate in supported_types:
                compute = candidate
                break
        else:
            compute = supported_types[0] if supported_types else "float16"

    return "cuda", compute, f"GPU acceleration active on {gpu_info.get('device_name')} with compute type {compute}"


def release_gpu_memory() -> None:
    """Trigger Python garbage collection and PyTorch CUDA cache clearing."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
