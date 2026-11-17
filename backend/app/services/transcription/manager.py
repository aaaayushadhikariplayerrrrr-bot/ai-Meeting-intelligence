"""ASR Engine Manager & Registry.

Manages the singleton lifecycle of the active speech recognition engine,
providing lazy initialization, resource sharing, and controlled unloading.
"""

import threading
import logging
from typing import Optional, Dict, Any

from app.core.config import settings
from app.services.transcription.base import ASREngine
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine

logger = logging.getLogger(__name__)

_engine_lock = threading.Lock()
_global_asr_engine: Optional[ASREngine] = None


def get_asr_engine(
    engine_name: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> ASREngine:
    """Retrieve or initialize the active speech recognition engine (singleton)."""
    global _global_asr_engine

    target_engine = engine_name or settings.ASR_ENGINE
    target_model = model_name or settings.ASR_MODEL_NAME
    target_device = device or settings.ASR_DEVICE

    with _engine_lock:
        if _global_asr_engine is None:
            logger.info(f"Initializing ASR engine '{target_engine}' with model '{target_model}'")
            if target_engine == "faster-whisper":
                _global_asr_engine = FasterWhisperEngine(
                    model_name=target_model,
                    device=target_device,
                    compute_type=settings.ASR_COMPUTE_TYPE,
                )
            else:
                raise ValueError(f"Unsupported ASR engine: {target_engine}")
        return _global_asr_engine


def unload_asr_engine() -> None:
    """Unload the active ASR engine to free GPU VRAM and system memory."""
    global _global_asr_engine
    with _engine_lock:
        if _global_asr_engine is not None:
            _global_asr_engine.unload_model()
            _global_asr_engine = None
            logger.info("Global ASR engine unloaded successfully.")
