"""Speaker diarization interface and provider abstraction.

CRITICAL DESIGN PRINCIPLE:
Whisper is an Automatic Speech Recognition (ASR) model, not a speaker diarization model.
This module defines a decoupled interface for future speaker diarization providers
(e.g., PyAnnote, WhisperX speaker clustering, or specialized Indian language diarizers)
without fabricating speaker identities or conflating ASR with diarization.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DiarizationSegment(BaseModel):
    """Time-stamped speaker turn identified by a diarization engine."""
    start_time: float
    end_time: float
    speaker_label: str = Field(..., description="Diarization cluster tag, e.g. 'SPEAKER_00'")
    confidence: Optional[float] = None


class DiarizationResult(BaseModel):
    """Result of speaker diarization."""
    turns: List[DiarizationSegment] = Field(default_factory=list)
    speaker_count: int = 0
    provider_name: str = "none"


class DiarizationProvider(ABC):
    """Abstract interface for speaker diarization engines."""

    @abstractmethod
    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """Perform speaker diarization on normalized audio."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether the diarization engine is configured and operational."""
        pass


class NullDiarizationProvider(DiarizationProvider):
    """Default provider for Phase 4: records that diarization was not executed.
    
    Guarantees that speaker identities are never invented or assumed from ASR alone.
    """

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        return DiarizationResult(
            turns=[],
            speaker_count=0,
            provider_name="none"
        )

    def is_available(self) -> bool:
        return False
