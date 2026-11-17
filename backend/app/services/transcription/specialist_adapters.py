"""Specialist ASR Engine Adapters (Stubs for Future Indian Language Models).

Prepared for future integration of:
1. AI4Bharat IndicConformer (optimized for 22 Indian languages including Hindi & Gujarati)
2. Oriserve Whisper-Hindi2Hinglish-Apex (fine-tuned for conversational code-switching)

CRITICAL DESIGN PRINCIPLE:
These models are defined with clean interface contracts conforming to `ASREngine`.
They are NOT installed automatically in Phase 4 to preserve environment stability,
bandwidth, and VRAM until full model evaluation in future phases.
"""

from typing import Optional, Dict, Any, Callable
from app.services.transcription.base import (
    ASREngine,
    ASRResult,
    TranscriptionOptions,
)


class IndicConformerEngine(ASREngine):
    """Adapter for AI4Bharat IndicConformer ASR.
    
    Targeted for Hindi, Gujarati, and other official Indian languages.
    """

    def __init__(self, model_version: str = "indicconformer-v1", device: str = "cuda"):
        self.model_version = model_version
        self.device = device
        self.is_loaded = False

    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        raise NotImplementedError(
            "AI4Bharat IndicConformer adapter will be enabled in a future phase after model licensing and evaluation."
        )

    def unload_model(self) -> None:
        self.is_loaded = False

    def is_gpu_available(self) -> bool:
        return False

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_name": "ai4bharat-indic-conformer",
            "status": "ready_for_evaluation",
            "supported_languages": ["hi", "gu", "mr", "bn", "ta", "te", "kn", "ml"],
            "installed": False,
        }

    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ASRResult:
        raise NotImplementedError("IndicConformer adapter is not yet installed.")


class WhisperHindi2HinglishApexEngine(ASREngine):
    """Adapter for Oriserve Whisper-Hindi2Hinglish-Apex.
    
    Targeted for conversational Hindi and Hinglish code-switching in corporate meetings.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.is_loaded = False

    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        raise NotImplementedError(
            "Whisper-Hindi2Hinglish-Apex adapter will be enabled in a future phase after model evaluation."
        )

    def unload_model(self) -> None:
        self.is_loaded = False

    def is_gpu_available(self) -> bool:
        return False

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_name": "oriserve-whisper-hindi2hinglish-apex",
            "status": "ready_for_evaluation",
            "supported_languages": ["hi", "hi-en", "en"],
            "installed": False,
        }

    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> ASRResult:
        raise NotImplementedError("Whisper-Hindi2Hinglish-Apex adapter is not yet installed.")
