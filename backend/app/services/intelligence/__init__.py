"""Structured Meeting Intelligence Extraction Service package."""

from .base import (
    ProviderType,
    IntelligenceError,
    SafeTranscriptRequiredError,
    IntelligenceDisabledError,
    ProviderExecutionError,
    IntelligenceExtractionResult,
)
from .engine import extract_meeting_intelligence, resolve_privacy_safe_transcript
from .deterministic import DeterministicExtractor
from .ollama_adapter import OllamaExtractor
from .gemini_adapter import GeminiExtractor
from .evidence import ground_and_verify_evidence
from .chunking import chunk_transcript_segments, merge_extracted_payloads
from .job_handler import process_extraction_job_sync
from .lineage import (
    LineageValidationError,
    CrossTenantLineageError,
    CircularLineageError,
    SelfSupersedingError,
    validate_decision_lineage,
    supersede_decision,
    record_commitment_review,
    resolve_unresolved_question,
    get_decision_lineage,
)
from .cross_meeting import (
    DecisionLifecycleService,
    CommitmentTrackingService,
    UnresolvedQuestionService,
)

__all__ = [
    "ProviderType",
    "IntelligenceError",
    "SafeTranscriptRequiredError",
    "IntelligenceDisabledError",
    "ProviderExecutionError",
    "IntelligenceExtractionResult",
    "extract_meeting_intelligence",
    "resolve_privacy_safe_transcript",
    "DeterministicExtractor",
    "OllamaExtractor",
    "GeminiExtractor",
    "ground_and_verify_evidence",
    "chunk_transcript_segments",
    "merge_extracted_payloads",
    "process_extraction_job_sync",
    "LineageValidationError",
    "CrossTenantLineageError",
    "CircularLineageError",
    "SelfSupersedingError",
    "validate_decision_lineage",
    "supersede_decision",
    "record_commitment_review",
    "resolve_unresolved_question",
    "get_decision_lineage",
    "DecisionLifecycleService",
    "CommitmentTrackingService",
    "UnresolvedQuestionService",
]
