"""Pydantic schemas and DTOs for Organizational Memory indexing jobs and RAG queries."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, AliasChoices


class IndexingJobPayload(BaseModel):
    """Payload stored in ProcessingJob for background memory indexing."""
    meeting_id: str = Field(..., min_length=1, description="UUID of the meeting to index")
    organization_id: str = Field(..., min_length=1, description="Tenant organization boundary UUID")
    transcript_id: Optional[str] = Field(
        default=None,
        description="Optional target privacy-safe transcript UUID. If omitted, selects the latest privacy-safe version."
    )
    force_reindex: bool = Field(
        default=False,
        description="If True, re-chunks and re-embeds even if already indexed."
    )

    model_config = ConfigDict(from_attributes=True)


class AbstentionStatus(str, Enum):
    """Explicit status indicating why an answer was not generated or how it was qualified."""
    NO_EVIDENCE = "NO_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    ANSWER_SUPPORTED = "ANSWER_SUPPORTED"
    NOT_ABSTAINED = "NOT_ABSTAINED"


class Citation(BaseModel):
    """Verified evidence citation strictly grounded in database transcript segments."""
    citation_id: str = Field(..., description="Unique citation label within the answer (e.g. 'cit-1')")
    chunk_id: str = Field(..., description="UUID of the supporting TranscriptChunkRecord")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    meeting_title: str = Field(..., description="Title of the parent meeting")
    meeting_date: Optional[str] = Field(default=None, description="ISO timestamp of the meeting")
    transcript_id: str = Field(..., description="UUID of the privacy-safe transcript")
    segment_ids: List[str] = Field(default_factory=list, description="Verified source TranscriptSegment UUIDs")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds from audio start")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds from audio start")
    quoted_text: str = Field(..., description="Relevant passage text quoted from the verified chunk")
    language_codes: List[str] = Field(default_factory=list, description="Language codes represented in snippet")

    model_config = ConfigDict(from_attributes=True)

    @property
    def segment_id(self) -> Optional[str]:
        return self.segment_ids[0] if self.segment_ids else self.chunk_id


class EvidenceChunk(BaseModel):
    """Evidence chunk for RAG context, compatible with test fixtures and runtime representations."""
    meeting_id: str
    segment_id: Optional[str] = None
    chunk_text: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    score: float = 0.0
    meeting_title: str = "Meeting"
    meeting_date: Optional[str] = None
    source_version: Optional[str] = None
    chunk_id: Optional[str] = None
    organization_id: Optional[str] = None
    segment_ids: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow", from_attributes=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.chunk_text and not self.text:
            self.text = self.chunk_text
        if self.segment_id and not self.chunk_id:
            self.chunk_id = self.segment_id
        if self.segment_id and not self.segment_ids:
            self.segment_ids = [self.segment_id]
        if self.chunk_id and not self.segment_ids:
            self.segment_ids = [self.chunk_id]


class EvidenceItem(BaseModel):
    """Structured evidence chunk retrieved from the organization's memory."""
    chunk_id: str = Field(..., description="UUID of the chunk record")
    organization_id: str = Field(..., description="Tenant organization boundary UUID")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    meeting_title: str = Field(..., description="Title of the parent meeting")
    meeting_date: Optional[str] = Field(default=None, description="ISO timestamp of the meeting")
    transcript_id: str = Field(..., description="UUID of the privacy-safe transcript")
    segment_ids: List[str] = Field(default_factory=list, description="Source transcript segment UUIDs")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds")
    language_codes: List[str] = Field(default_factory=list, description="Languages represented in chunk")
    dominant_language: str = Field(default="en", description="Dominant language code")
    text: str = Field(..., description="Privacy-safe chunk text")
    rrf_score: float = Field(..., ge=0.0, description="Reciprocal Rank Fusion score")
    keyword_rank: Optional[int] = Field(default=None, description="Rank from FTS5 keyword search")
    semantic_rank: Optional[int] = Field(default=None, description="Rank from dense vector search")

    model_config = ConfigDict(from_attributes=True)


class DecisionConflictItem(BaseModel):
    """Potential contradiction detected between active decisions."""
    decision_a_id: str = Field(..., description="UUID of the first decision")
    decision_a_text: str = Field(..., description="Description of the first decision")
    decision_a_meeting_title: str = Field(..., description="Meeting title of first decision")
    decision_a_date: Optional[str] = Field(default=None, description="Meeting date of first decision")
    decision_b_id: str = Field(..., description="UUID of the second decision")
    decision_b_text: str = Field(..., description="Description of the second decision")
    decision_b_meeting_title: str = Field(..., description="Meeting title of second decision")
    decision_b_date: Optional[str] = Field(default=None, description="Meeting date of second decision")
    subject: str = Field(..., description="Subject or technology topic where potential conflict exists")
    reason: str = Field(..., description="Reason why potential contradiction was detected")
    is_heuristic: bool = Field(default=True, description="True since conflict was detected via deterministic heuristic")
    suggested_action: str = Field(..., description="Recommended user resolution action")

    model_config = ConfigDict(from_attributes=True)


class TranscriptConflictItem(BaseModel):
    """Potential contradiction detected between transcript passages across meetings."""
    chunk_a_id: str = Field(..., description="UUID of the affirmative/first supporting chunk")
    chunk_a_text: str = Field(..., description="Text passage from the first meeting")
    chunk_a_meeting_title: str = Field(..., description="Meeting title of first passage")
    chunk_a_date: Optional[str] = Field(default=None, description="Meeting date of first passage")
    chunk_b_id: str = Field(..., description="UUID of the negative/second supporting chunk")
    chunk_b_text: str = Field(..., description="Text passage from the second meeting")
    chunk_b_meeting_title: str = Field(..., description="Meeting title of second passage")
    chunk_b_date: Optional[str] = Field(default=None, description="Meeting date of second passage")
    subject: str = Field(..., description="Topic or keyword where potential contradiction exists")
    reason: str = Field(..., description="Reason why potential contradiction was detected")
    is_heuristic: bool = Field(default=True, description="True since conflict was detected via deterministic heuristic")
    suggested_action: str = Field(
        default="Review the cited meeting segments to determine whether the policy was updated or remains disputed.",
        description="Recommended user resolution action"
    )

    model_config = ConfigDict(from_attributes=True)


class RetrievalDiagnostics(BaseModel):
    """Diagnostics and transparency metrics for RAG retrieval execution."""
    query: str = Field(..., description="Original user query")
    total_candidates_retrieved: int = Field(default=0, ge=0, description="Total candidates from hybrid retrieval")
    evidence_items_count: int = Field(default=0, ge=0, description="Number of evidence chunks assembled in context")
    meetings_represented: List[str] = Field(default_factory=list, description="Meeting IDs represented in evidence")
    provider_used: str = Field(default="deterministic", description="Identifier of the answer provider used")
    is_heuristic: bool = Field(default=True, description="True if answer was generated by deterministic fallback")
    fallback_warning: Optional[str] = Field(default=None, description="Warning if fallback was used")
    structured_decisions_count: int = Field(default=0, ge=0, description="Number of structured decisions retrieved")
    structured_commitments_count: int = Field(default=0, ge=0, description="Number of structured commitments retrieved")
    structured_questions_count: int = Field(default=0, ge=0, description="Number of structured questions retrieved")
    conflicts_detected_count: int = Field(default=0, ge=0, description="Number of potential decision conflicts detected")

    model_config = ConfigDict(from_attributes=True)


class RawAnswerResult(BaseModel):
    """Internal container returned by an AnswerProvider before citation verification."""
    answer_text: str = Field(..., description="Generated answer text")
    cited_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs referenced in answer")
    abstention_status: Optional[AbstentionStatus] = Field(default=None, description="Abstention status if applicable")
    abstention_reason: Optional[str] = Field(default=None, description="Human-readable reason for abstention")
    provider_used: str = Field(default="deterministic", description="Provider identifier")
    is_heuristic: bool = Field(default=True, description="True if heuristic/deterministic")
    fallback_warning: Optional[str] = Field(default=None, description="Optional fallback warning")
    structured_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Associated decisions")
    structured_commitments: List[Dict[str, Any]] = Field(default_factory=list, description="Associated commitments")
    structured_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Associated questions")
    detected_conflicts: List[DecisionConflictItem] = Field(default_factory=list, description="Detected conflicts")
    transcript_conflicts: List[TranscriptConflictItem] = Field(default_factory=list, description="Detected transcript contradictions")

    model_config = ConfigDict(from_attributes=True)


class RAGQueryRequest(BaseModel):
    """Request DTO for cross-meeting RAG question answering."""
    question: str = Field(..., min_length=1, max_length=1000, description="Natural-language question")
    organization_id: str = Field(..., min_length=1, description="Tenant organization boundary UUID")
    meeting_id: Optional[str] = Field(default=None, description="Optional meeting UUID to scope search to a single meeting")
    start_date: Optional[datetime] = Field(default=None, description="Optional earliest meeting date filter")
    end_date: Optional[datetime] = Field(default=None, description="Optional latest meeting date filter")
    max_evidence_chunks: int = Field(default=5, ge=1, le=20, description="Maximum evidence chunks in assembled context")
    output_language: str = Field(default="en", description="Target output language code (e.g. 'en', 'hi', 'gu')")

    model_config = ConfigDict(from_attributes=True)


class RAGQueryResponse(BaseModel):
    """Response DTO for cross-meeting RAG question answering."""
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded factual answer or abstention notice")
    citations: List[Citation] = Field(default_factory=list, description="Verified evidence citations")
    abstention_status: Optional[AbstentionStatus] = Field(default=None, description="Abstention category if abstained")
    is_abstained: bool = Field(default=False, description="True if the engine abstained from answering")
    diagnostics: RetrievalDiagnostics = Field(..., description="Retrieval and provider diagnostics")
    structured_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant structured decisions")
    structured_commitments: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant structured commitments")
    structured_questions: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant structured questions")
    detected_conflicts: List[DecisionConflictItem] = Field(default_factory=list, description="Potentially contradictory active decisions")
    transcript_conflicts: List[TranscriptConflictItem] = Field(default_factory=list, description="Potentially contradictory transcript passages across meetings")

    model_config = ConfigDict(from_attributes=True)


class MemoryQueryApiRequest(BaseModel):
    """User request for querying cross-meeting organizational intelligence."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language question to ask organizational memory",
        examples=["What container orchestration system was adopted and why?"]
    )
    meeting_id: Optional[str] = Field(
        default=None,
        description="Optional meeting UUID to scope the question to a single meeting"
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Optional earliest meeting date boundary (ISO 8601)"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="Optional latest meeting date boundary (ISO 8601)"
    )
    max_evidence_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of evidence chunks to assemble into context",
        validation_alias=AliasChoices("max_evidence_chunks", "max_chunks"),
    )
    output_language: str = Field(
        default="en",
        max_length=10,
        description="Target output language code (e.g. 'en', 'hi', 'gu')"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Client-supplied organization ID (strictly ignored; server uses authenticated context)"
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("question")
    @classmethod
    def validate_question_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Question cannot be empty or whitespace-only.")
        return s

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be later than end_date.")
        return self


class MemorySearchApiRequest(BaseModel):
    """User request for hybrid search across organizational memory chunks."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Keyword or semantic search query",
        examples=["Kubernetes production deployment"]
    )
    meeting_id: Optional[str] = Field(
        default=None,
        description="Optional meeting UUID to scope search"
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Optional earliest meeting date boundary (ISO 8601)"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="Optional latest meeting date boundary (ISO 8601)"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of search results to return",
        validation_alias=AliasChoices("limit", "top_k"),
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Client-supplied organization ID (strictly ignored; server uses authenticated context)"
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("query")
    @classmethod
    def validate_query_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Query cannot be empty or whitespace-only.")
        return s

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be later than end_date.")
        return self


class MemorySearchResultItem(BaseModel):
    """Individual search result item representing a privacy-safe transcript chunk."""
    chunk_id: str = Field(..., description="UUID of the supporting TranscriptChunkRecord")
    meeting_id: str = Field(..., description="UUID of the parent meeting")
    meeting_title: str = Field(..., description="Title of the parent meeting")
    meeting_date: Optional[str] = Field(default=None, description="ISO timestamp of the meeting")
    transcript_id: str = Field(..., description="UUID of the privacy-safe transcript")
    segment_ids: List[str] = Field(default_factory=list, description="Verified source TranscriptSegment UUIDs")
    start_time: float = Field(..., ge=0.0, description="Start timestamp in seconds from audio start")
    end_time: float = Field(..., ge=0.0, description="End timestamp in seconds from audio start")
    text: str = Field(..., description="Privacy-safe chunk text passage")
    language_codes: List[str] = Field(default_factory=list, description="Language codes represented in snippet")
    dominant_language: str = Field(default="en", description="Dominant language code")
    rrf_score: float = Field(default=0.0, ge=0.0, description="Reciprocal Rank Fusion score")
    keyword_rank: Optional[int] = Field(default=None, description="Rank from FTS5 keyword search")
    semantic_rank: Optional[int] = Field(default=None, description="Rank from dense vector search")

    model_config = ConfigDict(from_attributes=True)


class MemorySearchResponse(BaseModel):
    """Response DTO for organizational memory hybrid search."""
    query: str = Field(..., description="Original user search query")
    total_results: int = Field(default=0, ge=0, description="Number of results returned")
    results: List[MemorySearchResultItem] = Field(default_factory=list, description="Matching transcript passages")

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "IndexingJobPayload",
    "AbstentionStatus",
    "Citation",
    "EvidenceItem",
    "DecisionConflictItem",
    "TranscriptConflictItem",
    "RetrievalDiagnostics",
    "RawAnswerResult",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "MemoryQueryApiRequest",
    "MemorySearchApiRequest",
    "MemorySearchResultItem",
    "MemorySearchResponse",
]
