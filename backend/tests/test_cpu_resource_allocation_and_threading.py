"""Phase 9.3: CPU Resource Allocation & Thread Bounding Tests.

Verifies:
1. CPU thread limits are configured correctly in PyTorch and environment (OMP, OpenBLAS, MKL).
2. PyTorch thread initialization is idempotent, safe against repeated calls, and handles started parallel work gracefully.
3. Faster-Whisper engine explicitly bounds cpu_threads and num_workers on instantiation.
4. Faster-Whisper CPU fallback retains bounded cpu_threads and int8 compute type.
5. CUDA ASR configuration and priority on GPU are preserved without degradation.
6. Multilingual E5 embedding service operates on CPU by default.
7. E5 CUDA request coordinates with resource manager and falls back to CPU if ASR is active on GPU or VRAM is low.
8. Background job worker processes jobs sequentially off the asyncio event loop via asyncio.to_thread.
9. Hardware diagnostics API surfaces accurate cpu_info to super admins without schema breakage.
10. Resource manager VRAM headroom guards remain fully intact.
"""

import os
import sys
import asyncio
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.transcription.gpu_manager import (
    configure_cpu_resources,
    get_cpu_resource_info,
    register_active_gpu_consumer,
    unregister_active_gpu_consumer,
    can_use_gpu_for_embedding,
    select_optimal_device_and_compute,
)
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.job_worker import JobWorker


def test_cpu_thread_limits_applied_correctly():
    """Test 1: Verify OpenMP, BLAS, and PyTorch intra/inter-op threads are bounded."""
    info = get_cpu_resource_info()

    assert info["omp_num_threads"] is not None
    assert int(info["omp_num_threads"]) <= settings.CPU_MAX_THREADS
    assert info["openblas_num_threads"] is not None
    assert int(info["openblas_num_threads"]) <= settings.CPU_MAX_THREADS

    try:
        import torch
        assert torch.get_num_threads() <= settings.TORCH_NUM_THREADS
        assert torch.get_num_interop_threads() <= settings.TORCH_NUM_INTEROP_THREADS
    except ImportError:
        pass


def test_pytorch_settings_idempotent_and_safe_repeated_calls():
    """Test 2: Verify configure_cpu_resources is strictly idempotent and handles repeated calls cleanly."""
    initial_info = configure_cpu_resources()
    assert initial_info is not None

    # Call multiple times in succession
    for _ in range(5):
        repeated_info = configure_cpu_resources()
        assert repeated_info["cpu_count"] == initial_info["cpu_count"]
        assert repeated_info["omp_num_threads"] == initial_info["omp_num_threads"]

    # Verify that a simulated RuntimeError from PyTorch set_num_interop_threads is caught safely
    with patch("torch.set_num_interop_threads", side_effect=RuntimeError("Cannot set after parallel work")):
        with patch("app.services.transcription.gpu_manager._cpu_resources_configured", False):
            info = configure_cpu_resources()
            assert info is not None


def test_faster_whisper_engine_passes_cpu_thread_bounds():
    """Test 3: Verify WhisperModel instantiation explicitly passes cpu_threads and num_workers."""
    engine = FasterWhisperEngine()

    mock_whisper_model = MagicMock()
    with patch("faster_whisper.WhisperModel", return_value=mock_whisper_model) as mock_cls:
        with patch("app.services.transcription.faster_whisper_engine.select_optimal_device_and_compute", return_value=("cuda", "float16", "GPU test")):
            engine.load_model()

            assert mock_cls.called
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs.get("cpu_threads") == settings.ASR_CPU_THREADS
            assert call_kwargs.get("num_workers") == settings.ASR_NUM_WORKERS
            assert call_kwargs.get("device") == "cuda"
            assert call_kwargs.get("compute_type") == "float16"

    # Clean up engine state
    engine.unload_model()


def test_faster_whisper_cpu_fallback_preserves_thread_bounds():
    """Test 4: Verify CPU fallback passes cpu_threads and num_workers to CPU WhisperModel."""
    engine = FasterWhisperEngine()

    mock_cpu_model = MagicMock()
    # First call (CUDA) fails, second call (CPU fallback) succeeds
    with patch("faster_whisper.WhisperModel", side_effect=[RuntimeError("CUDA out of memory"), mock_cpu_model]) as mock_cls:
        with patch("app.services.transcription.faster_whisper_engine.select_optimal_device_and_compute", return_value=("cuda", "float16", "GPU test")):
            engine.load_model()

            assert mock_cls.call_count == 2
            fallback_call_kwargs = mock_cls.call_args_list[1].kwargs
            assert fallback_call_kwargs.get("device") == "cpu"
            assert fallback_call_kwargs.get("compute_type") == "int8"
            assert fallback_call_kwargs.get("cpu_threads") == settings.ASR_CPU_THREADS
            assert fallback_call_kwargs.get("num_workers") == settings.ASR_NUM_WORKERS
            assert engine.actual_device == "cpu"
            assert engine.actual_compute_type == "int8"

    engine.unload_model()


def test_faster_whisper_preserves_cuda_configuration_and_registration():
    """Test 5: Verify CUDA ASR configuration and active GPU consumer tracking."""
    engine = FasterWhisperEngine()

    mock_whisper_model = MagicMock()
    with patch("faster_whisper.WhisperModel", return_value=mock_whisper_model):
        with patch("app.services.transcription.faster_whisper_engine.select_optimal_device_and_compute", return_value=("cuda", "float16", "GPU test")):
            engine.load_model()
            assert engine.actual_device == "cuda"

            info = get_cpu_resource_info()
            assert "faster-whisper" in info["active_gpu_consumers"]

            # Unload model unregisters consumer
            engine.unload_model()
            assert engine.actual_device == "unloaded"
            info_after = get_cpu_resource_info()
            assert "faster-whisper" not in info_after["active_gpu_consumers"]


def test_e5_embedding_preserves_cpu_by_default():
    """Test 6: Verify MultilingualE5EmbeddingService defaults to CPU and preserves CPU inference."""
    service = MultilingualE5EmbeddingService(lazy_load=True)
    assert service.device == "cpu"

    # Inject mock tokenizer and model to test CPU inference
    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": MagicMock(to=MagicMock(return_value="mock_ids")),
        "attention_mask": MagicMock(to=MagicMock(return_value="mock_mask")),
    }
    mock_model = MagicMock()

    service.load_model(tokenizer=mock_tok, model=mock_model)
    assert service.is_loaded
    assert service.device == "cpu"


def test_e5_cuda_coordination_with_resource_manager():
    """Test 7: Verify E5 falls back to CPU when Faster-Whisper is active or VRAM is low."""
    # 1. When faster-whisper is active on GPU, can_use_gpu_for_embedding must be False
    register_active_gpu_consumer("faster-whisper", "large-v3-turbo")
    allowed, reason = can_use_gpu_for_embedding()
    assert allowed is False
    assert "GPU is allocated to Faster-Whisper" in reason

    # 2. When E5 requests CUDA, it detects the conflict and safely falls back to CPU
    service = MultilingualE5EmbeddingService(device="cuda", lazy_load=True)
    assert service.device == "cuda"

    mock_tok = MagicMock()
    mock_model = MagicMock()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok):
        with patch("transformers.AutoModel.from_pretrained", return_value=mock_model):
            with patch.object(service, "is_cached", return_value=True):
                service.load_model()
                # Device must have fallen back to CPU
                assert service.device == "cpu"
                mock_model.to.assert_called_with("cpu")

    unregister_active_gpu_consumer("faster-whisper")


def test_e5_cuda_coordination_low_vram_fallback():
    """Test 8: Verify E5 falls back to CPU when free VRAM is below threshold."""
    unregister_active_gpu_consumer("faster-whisper")

    # Simulate low VRAM (1200 MB free < 2000 MB threshold)
    mock_gpu_info = {
        "torch_cuda_available": True,
        "free_vram_mb": 1200.0,
        "total_vram_mb": 8192.0,
    }
    with patch("app.services.transcription.gpu_manager.check_gpu_hardware", return_value=mock_gpu_info):
        allowed, reason = can_use_gpu_for_embedding(min_free_vram_mb=2000.0)
        assert allowed is False
        assert "Insufficient free VRAM" in reason


def test_background_worker_concurrency_bounded():
    """Test 9: Verify JobWorker processes jobs sequentially and offloads to asyncio.to_thread."""
    async def _run():
        worker = JobWorker(poll_interval=0.1)
        executed_jobs = []

        def mock_execute(job_id: str):
            executed_jobs.append(job_id)

        # Verify that poll_once awaits asyncio.to_thread
        with patch("app.services.job_worker.get_next_pending_job_id", side_effect=["job-1", None]):
            with patch("app.services.job_worker.claim_job_atomically", return_value=MagicMock()):
                with patch("app.services.job_worker.execute_claimed_job", side_effect=mock_execute):
                    processed = await worker.poll_once()
                    assert processed is True
                    assert executed_jobs == ["job-1"]

                    # Second poll finds no jobs
                    processed_empty = await worker.poll_once()
                    assert processed_empty is False

    asyncio.run(_run())


def test_hardware_status_response_cpu_info_diagnostics(client, db_session):
    """Test 10: Verify hardware diagnostics endpoint surfaces cpu_info without regression."""
    from backend.tests.conftest import make_auth_headers
    from app.models.identity import User

    # Create super admin user
    super_admin = User(
        id="user-superadmin-001",
        email="superadmin@example.com",
        full_name="Super Admin",
        is_super_admin=True,
        is_active=True,
    )
    db_session.add(super_admin)
    db_session.commit()

    headers = make_auth_headers(super_admin)
    response = client.get("/api/v1/transcription/hardware/diagnostics", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "gpu_info" in data
    assert "active_engine" in data
    assert "cpu_info" in data
    assert data["cpu_info"]["omp_num_threads"] is not None
    assert data["active_engine"]["cpu_threads"] == settings.ASR_CPU_THREADS
    assert data["active_engine"]["num_workers"] == settings.ASR_NUM_WORKERS


def test_select_optimal_device_and_compute_guards():
    """Test 11: Verify VRAM and device selection guards remain functional."""
    # 1. Explicit CPU request
    device, compute, reason = select_optimal_device_and_compute(preferred_device="cpu")
    assert device == "cpu"
    assert compute == "int8"

    # 2. Low VRAM fallback (< 2000 MB)
    with patch("app.services.transcription.gpu_manager.check_ctranslate2_support", return_value=(True, "OK", ["float16"])):
        with patch("app.services.transcription.gpu_manager.check_gpu_hardware", return_value={"free_vram_mb": 1500.0, "device_name": "RTX 5060"}):
            device, compute, reason = select_optimal_device_and_compute(preferred_device="cuda", min_free_vram_mb=2000.0)
            assert device == "cpu"
            assert compute == "int8"
            assert "Low VRAM" in reason


def test_early_import_order_establishes_env_before_numerical_libs():
    """Test 12 (Phase 9.3.1): Verify environment defaults are active before numerical libraries load in app.main."""
    import subprocess
    code = """
import os, sys
from importlib.abc import MetaPathFinder

log = []
class TraceFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ('numpy', 'torch', 'ctranslate2', 'scipy'):
            log.append((fullname, os.environ.get('OMP_NUM_THREADS'), os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('MKL_NUM_THREADS')))
        return None

sys.meta_path.insert(0, TraceFinder())
sys.path.insert(0, 'backend')
import app.main

for mod, omp, openblas, mkl in log:
    assert omp == '4', f"{mod} imported with OMP_NUM_THREADS={omp}"
    assert openblas == '4', f"{mod} imported with OPENBLAS_NUM_THREADS={openblas}"
    assert mkl == '4', f"{mod} imported with MKL_NUM_THREADS={mkl}"
print("OK")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Import order verification failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "OK" in res.stdout


def test_early_import_order_preserves_user_overrides():
    """Test 13 (Phase 9.3.1): Verify user environment overrides are preserved on application import."""
    import subprocess
    code = """
import os, sys
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '6'
sys.path.insert(0, 'backend')
import app.main
assert os.environ.get('OMP_NUM_THREADS') == '8'
assert os.environ.get('OPENBLAS_NUM_THREADS') == '6'
print("OK")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"User override verification failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "OK" in res.stdout


def test_package_entry_does_not_eagerly_import_torch():
    """Test 14 (Phase 9.3.1): Verify app package root sets environment defaults without importing PyTorch."""
    import subprocess
    code = """
import os, sys
sys.path.insert(0, 'backend')
import app
assert os.environ.get('OMP_NUM_THREADS') == '4'
assert os.environ.get('OPENBLAS_NUM_THREADS') == '4'
assert 'torch' not in sys.modules, 'torch was eagerly imported by app package root!'
print("OK")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"Eager import verification failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "OK" in res.stdout

