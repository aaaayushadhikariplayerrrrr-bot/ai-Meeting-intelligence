"""Base types, exceptions, and data models for the Intelligence Extraction service."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from app.schemas.intelligence import (
    MeetingSummaryResponse,
    TopicResponse,
    DecisionResponse,
    ActionItemResponse,
    RiskResponse,
    UnresolvedQuestionResponse,
    StructuredExtractionPayload,
)


class ProviderType(str, Enum):
    """Supported intelligence extraction engine providers."""
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    OLLAMA = "ollama"


class IntelligenceError(Exception):
    """Base exception for intelligence extraction failures."""
    pass


class SafeTranscriptRequiredError(IntelligenceError):
    """Raised when an unredacted or unsafe transcript is provided for intelligence extraction."""
    pass


class IntelligenceDisabledError(IntelligenceError):
    """Raised when intelligence extraction is requested while the provider is disabled."""
    pass


class ProviderExecutionError(IntelligenceError):
    """Raised when an intelligence provider fails to execute or produce valid output."""
    pass


@dataclass
class IntelligenceExtractionResult:
    """Standardized result returned by the intelligence extraction service."""
    meeting_id: str
    source_transcript_id: str
    summary: Optional[MeetingSummaryResponse] = None
    topics: List[TopicResponse] = field(default_factory=list)
    decisions: List[DecisionResponse] = field(default_factory=list)
    action_items: List[ActionItemResponse] = field(default_factory=list)
    risks: List[RiskResponse] = field(default_factory=list)
    unresolved_questions: List[UnresolvedQuestionResponse] = field(default_factory=list)
    provider_used: str = "deterministic"
    is_heuristic: bool = False
    fallback_warning: Optional[str] = None
    is_partial: bool = False
    partial_reason: Optional[str] = None
    raw_payload: Optional[StructuredExtractionPayload] = None
