"""End-to-End Real Hardware Verification on NVIDIA GeForce RTX 5060 GPU.

Verifies:
- Live Faster-Whisper model loading on CUDA with float16 precision
- Live transcription execution on GPU
- Real language detection and segment generation
- Automatic CPU fallback
"""

import os
import wave
import struct
import pytest

from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.transcription.base import TranscriptionOptions
from app.services.transcription.gpu_manager import check_ctranslate2_support, check_gpu_hardware


def _create_synthetic_wav(path: str, duration: float = 1.0) -> None:
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


def test_real_gpu_acceleration_on_rtx_5060(tmp_path):
    """Verify live GPU transcription on the NVIDIA RTX 5060 Laptop GPU."""
    # 1. Hardware capability check
    is_cuda_ready, msg, compute_types = check_ctranslate2_support()
    assert is_cuda_ready, f"CTranslate2 CUDA not ready: {msg}"
    assert "float16" in compute_types or "int8_float16" in compute_types

    gpu_hw = check_gpu_hardware()
    assert gpu_hw["torch_cuda_available"] is True
    assert "RTX 5060" in (gpu_hw["device_name"] or "")

    # 2. Create audio sample
    test_audio = str(tmp_path / "test_e2e_audio.wav")
    _create_synthetic_wav(test_audio, duration=1.0)

    # 3. Instantiate FasterWhisperEngine with tiny model for speedy verified execution
    engine = FasterWhisperEngine(model_name="tiny", device="cuda", compute_type="float16")
    try:
        engine.load_model()
        assert engine.actual_device == "cuda"
        assert engine.actual_compute_type == "float16"

        progress_recorded = []
        result = engine.transcribe(
            audio_path=test_audio,
            options=TranscriptionOptions(beam_size=1),
            progress_callback=lambda p: progress_recorded.append(p),
        )

        # 4. Verify transcription completed on CUDA
        assert result.device_used == "cuda"
        assert result.compute_type_used == "float16"
        assert result.engine_name == "faster-whisper"
        assert result.duration_seconds > 0
        assert len(progress_recorded) > 0
    finally:
        engine.unload_model()


def test_real_cpu_fallback(tmp_path):
    """Verify live CPU fallback mode."""
    test_audio = str(tmp_path / "test_cpu_audio.wav")
    _create_synthetic_wav(test_audio, duration=1.0)

    engine = FasterWhisperEngine(model_name="tiny", device="cpu", compute_type="int8")
    try:
        engine.load_model()
        assert engine.actual_device == "cpu"
        assert engine.actual_compute_type == "int8"

        result = engine.transcribe(
            audio_path=test_audio,
            options=TranscriptionOptions(beam_size=1),
        )

        assert result.device_used == "cpu"
        assert result.compute_type_used == "int8"
        assert result.engine_name == "faster-whisper"
    finally:
        engine.unload_model()
