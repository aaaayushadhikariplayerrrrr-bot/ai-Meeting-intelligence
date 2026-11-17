from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel, Field


class ASRSegment(BaseModel):
    """Normalized segment representation from an ASR engine."""
    sequence_order: int
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed text in original spoken language")
    language_code: str = Field(..., description="ISO 639-1 language code (e.g. 'en', 'hi', 'gu')")
    confidence: Optional[float] = Field(
        None,
        description="Uncalibrated confidence score derived from exp(avg_logprob), bounded [0.0, 1.0]. Not a statistically calibrated posterior probability."
    )
    avg_logprob: Optional[float] = Field(
        None,
        description="Raw average token log probability from ASR decoder (uncalibrated model score, typically <= 0.0)"
    )
    speaker_label: Optional[str] = Field(None, description="Speaker identifier if diarized (e.g. 'SPEAKER_00')")
    words: Optional[List[Dict[str, Any]]] = Field(None, description="Optional word-level timestamps")


class ASRResult(BaseModel):
    """Complete result returned by an ASR engine transcription run."""
    segments: List[ASRSegment] = Field(default_factory=list)
    full_text: str = ""
    detected_language: Optional[str] = Field(None, description="Dominant language detected by model")
    language_probability: Optional[float] = Field(None, description="Confidence of language detection (0.0 to 1.0)")
    duration_seconds: float = Field(0.0, description="Total processed audio duration in seconds")
    device_used: str = Field("unknown", description="Actual execution device ('cuda:0', 'cpu')")
    compute_type_used: str = Field("unknown", description="Actual compute precision ('float16', 'int8', etc.)")
    model_name: str = Field("", description="Model identifier used")
    engine_name: str = Field("", description="ASR engine provider name")
    transcription_mode: str = Field(
        "standard",
        description="Application transcription mode ('standard' or 'hinglish')"
    )


class TranscriptionOptions(BaseModel):
    """Configuration options passed to an ASR transcription run."""
    language: Optional[str] = Field(
        None,
        description="Optional forced ISO language code (e.g. 'en', 'hi', 'gu'). None for auto-detection."
    )
    transcription_mode: str = Field(
        "standard",
        description="Application transcription mode: 'standard' or 'hinglish' (Hindi-English code-switching)"
    )
    initial_prompt: Optional[str] = Field(
        None,
        description="Optional prompt text to condition the decoder (e.g. for Hinglish code-switching or vocabulary)"
    )
    beam_size: int = Field(5, description="Beam search width")
    temperature: float = Field(0.0, description="Sampling temperature (0.0 for greedy/deterministic)")
    vad_filter: bool = Field(True, description="Enable Voice Activity Detection to filter non-speech")
    vad_min_silence_duration_ms: int = Field(500, description="VAD min silence duration in milliseconds")
    chunk_length_seconds: Optional[float] = Field(300.0, description="Chunk length for long audio (None to disable)")
    chunk_overlap_seconds: float = Field(2.0, description="Overlap between chunks in seconds")


class ASREngine(ABC):
    """Abstract Base Class for model-independent ASR providers."""

    @abstractmethod
    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        """Load the ASR model onto the specified or default device."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free GPU / system memory."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        options: Optional[TranscriptionOptions] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> ASRResult:
        """Transcribe an audio file and return normalized segments and metadata."""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Return engine metadata, active device, and supported capabilities."""
        pass

    @abstractmethod
    def is_gpu_available(self) -> bool:
        """Check whether GPU acceleration is available and operational."""
        pass
