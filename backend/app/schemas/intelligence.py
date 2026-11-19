"""Pydantic schemas for structured meeting intelligence.

Covers:
- Meeting summaries (executive and detailed)
- Topics and agenda discussion points
- Decisions with rationale and evidence grounding
- Action items with assignees, deadlines, and lifecycle statuses
- Identified risks, severity ratings, and mitigations
- Unresolved follow-up questions
- Combined intelligence responses and extraction job DTOs
- Status update payloads
"""

from datetime import datetime
from typing import List, Optional, Union, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.intelligence import DecisionStatus, ActionItemStatus, RiskSeverity, QuestionStatus


# --- Creation / Validation DTOs (used by extraction services) ---

class TopicCreate(BaseModel):
    """DTO for creating or extracting a meeting topic."""
    title: str = Field(..., min_length=1, max_length=255, description="Clear topic or agenda title")
    description: Optional[str] = Field(None, description="Concise synopsis of the discussion on this topic")
    start_time: Optional[float] = Field(None, ge=0.0, description="Discussion start time in seconds")
    end_time: Optional[float] = Field(None, ge=0.0, description="Discussion end time in seconds")

    @model_validator(mode="after")
    def validate_time_range(self) -> "TopicCreate":
        if self.start_time is not None and self.end_time is not None:
            if self.end_time < self.start_time:
                raise ValueError("end_time cannot be earlier than start_time.")
        return self

    @property
    def name(self) -> str:
        return self.title


class DecisionCreate(BaseModel):
    """DTO for creating or extracting an agreed decision."""
    description: str = Field(..., min_length=1, description="Explicit statement of the decision reached")
    rationale_or_context: Optional[str] = Field(None, description="Why this decision was made or alternative options considered")
    decided_by: Optional[str] = Field(None, max_length=255, description="Individual, group, or team responsible for the decision")
    evidence_segment_id: Optional[str] = Field(None, description="Foreign key to the supporting transcript segment")
    timestamp_seconds: Optional[float] = Field(None, ge=0.0, description="Exact timeline second where decision was finalized")
    status: DecisionStatus = Field(DecisionStatus.ACTIVE, description="Lifecycle status of the decision")
    supersedes_decision_id: Optional[str] = Field(None, description="UUID of earlier decision superseded by this one")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return DecisionStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid decision status '{v}'. Allowed values: {[s.value for s in DecisionStatus]}"
                )
        return v

    @property
    def title(self) -> str:
        return self.description

    @property
    def rationale(self) -> Optional[str]:
        return self.rationale_or_context


class ActionItemCreate(BaseModel):
    """DTO for creating or extracting a committed action item."""
    description: str = Field(..., min_length=1, description="Committed task description")
    owner_name: Optional[str] = Field(None, max_length=255, description="Name of individual or team owning the task")
    owner_user_id: Optional[str] = Field(None, description="Optional system user ID if owner is an organization member")
    deadline: Optional[datetime] = Field(None, description="Explicit or inferred deadline timestamp")
    status: ActionItemStatus = Field(ActionItemStatus.PENDING, description="Initial status of the action item")
    evidence_segment_id: Optional[str] = Field(None, description="Foreign key to the supporting transcript segment")
    timestamp_seconds: Optional[float] = Field(None, ge=0.0, description="Exact timeline second where action was assigned")
    last_reviewed_meeting_id: Optional[str] = Field(None, description="Meeting where this commitment was reviewed/updated")
    last_evidence_segment_id: Optional[str] = Field(None, description="Transcript segment providing evidence for the review")
    last_reviewed_at: Optional[datetime] = Field(None, description="Timestamp when this commitment was reviewed")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return ActionItemStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid action item status '{v}'. Allowed values: {[s.value for s in ActionItemStatus]}"
                )
        return v

    @property
    def assignee(self) -> Optional[str]:
        return self.owner_name


class RiskCreate(BaseModel):
    """DTO for creating or extracting an identified risk or blocker."""
    description: str = Field(..., min_length=1, description="Description of the risk, impediment, or concern")
    severity: RiskSeverity = Field(RiskSeverity.MEDIUM, description="Assessed severity level")
    mitigation_suggestion: Optional[str] = Field(None, description="Proposed or agreed mitigation approach")
    evidence_segment_id: Optional[str] = Field(None, description="Foreign key to the supporting transcript segment")

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return RiskSeverity(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid risk severity '{v}'. Allowed values: {[s.value for s in RiskSeverity]}"
                )
        return v

    @property
    def mitigation(self) -> Optional[str]:
        return self.mitigation_suggestion


class UnresolvedQuestionCreate(BaseModel):
    """DTO for creating or extracting an unresolved question."""
    question_text: str = Field(..., min_length=1, description="The open question or topic requiring future resolution")
    raised_by: Optional[str] = Field(None, max_length=255, description="Speaker or participant who raised the question")
    status: QuestionStatus = Field(QuestionStatus.OPEN, description="Status of the question")
    evidence_segment_id: Optional[str] = Field(None, description="Foreign key to the supporting transcript segment")
    resolved_at: Optional[datetime] = Field(None, description="Timestamp when this question was resolved")
    resolving_meeting_id: Optional[str] = Field(None, description="Meeting where question was resolved")
    resolving_decision_id: Optional[str] = Field(None, description="Decision that resolved this question")
    resolution_notes: Optional[str] = Field(None, description="Explanation of how the question was resolved")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return QuestionStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid question status '{v}'. Allowed values: {[s.value for s in QuestionStatus]}"
                )
        return v

    @property
    def question(self) -> str:
        return self.question_text


class MeetingSummaryCreate(BaseModel):
    """DTO for creating or updating a meeting summary record."""
    executive_summary: str = Field(..., min_length=1, description="Concise executive synthesis (1-2 paragraphs)")
    detailed_summary: Optional[str] = Field(None, description="Comprehensive topic-by-topic discussion summary")
    key_takeaways: Optional[str] = Field(None, description="Bullet points or JSON serialized list of key takeaways")
    version: int = Field(1, ge=1, description="Version number of this summary derivation")


class StructuredExtractionPayload(BaseModel):
    """Intermediate schema for validating raw LLM or NLP extraction output before DB persistence."""
    executive_summary: str = Field(..., min_length=1)
    detailed_summary: Optional[str] = None
    key_takeaways: Optional[List[str]] = None
    topics: List[TopicCreate] = Field(default_factory=list)
    decisions: List[DecisionCreate] = Field(default_factory=list)
    action_items: List[ActionItemCreate] = Field(default_factory=list)
    risks: List[RiskCreate] = Field(default_factory=list)
    unresolved_questions: List[UnresolvedQuestionCreate] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.executive_summary


# --- API Response DTOs (from_attributes=True for ORM serialization) ---

class TopicResponse(BaseModel):
    """Response schema for a meeting topic."""
    id: str
    meeting_id: str
    title: str
    description: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DecisionResponse(BaseModel):
    """Response schema for an agreed decision with evidence grounding and lifecycle tracking."""
    id: str
    meeting_id: str
    description: str
    rationale_or_context: Optional[str] = None
    decided_by: Optional[str] = None
    evidence_segment_id: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    status: str = DecisionStatus.ACTIVE.value
    supersedes_decision_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActionItemResponse(BaseModel):
    """Response schema for a committed action item."""
    id: str
    meeting_id: str
    description: str
    owner_name: Optional[str] = None
    owner_user_id: Optional[str] = None
    deadline: Optional[datetime] = None
    status: str
    evidence_segment_id: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    last_reviewed_meeting_id: Optional[str] = None
    last_evidence_segment_id: Optional[str] = None
    last_reviewed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RiskResponse(BaseModel):
    """Response schema for an identified risk or blocker."""
    id: str
    meeting_id: str
    description: str
    severity: str
    mitigation_suggestion: Optional[str] = None
    evidence_segment_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UnresolvedQuestionResponse(BaseModel):
    """Response schema for an open, deferred, or resolved question."""
    id: str
    meeting_id: str
    question_text: str
    raised_by: Optional[str] = None
    status: str
    evidence_segment_id: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolving_meeting_id: Optional[str] = None
    resolving_decision_id: Optional[str] = None
    resolution_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActionItemReviewResponse(BaseModel):
    """Response schema for a single review event of an action item."""
    id: str
    action_item_id: str
    meeting_id: str
    meeting_title: Optional[str] = None
    evidence_segment_id: Optional[str] = None
    evidence_quote: Optional[str] = None
    reviewer_user_id: Optional[str] = None
    status_at_review: Optional[str] = None
    review_notes: Optional[str] = None
    reviewed_at: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CrossMeetingDecisionResponse(DecisionResponse):
    """Extended decision response for cross-meeting listings and lineage display."""
    meeting_title: Optional[str] = None
    meeting_date: Optional[datetime] = None
    superseded_by_decision_id: Optional[str] = None
    lineage_depth: int = 0
    lineage: List[DecisionResponse] = Field(default_factory=list)


class CrossMeetingCommitmentResponse(ActionItemResponse):
    """Extended commitment response including cross-meeting tracking and review history."""
    origin_meeting_title: Optional[str] = None
    origin_meeting_date: Optional[datetime] = None
    last_reviewed_meeting_title: Optional[str] = None
    is_overdue: bool = False
    review_count: int = 0
    reviews: List[ActionItemReviewResponse] = Field(default_factory=list)


class CrossMeetingQuestionResponse(UnresolvedQuestionResponse):
    """Extended unresolved question response including resolving meeting and decision context."""
    origin_meeting_title: Optional[str] = None
    origin_meeting_date: Optional[datetime] = None
    resolving_meeting_title: Optional[str] = None
    resolving_decision_description: Optional[str] = None


class MeetingSummaryResponse(BaseModel):
    """Response schema for meeting summaries."""
    id: str
    meeting_id: str
    transcript_id: Optional[str] = None
    executive_summary: str
    detailed_summary: Optional[str] = None
    key_takeaways: Optional[str] = None
    version: int = 1
    provider_used: Optional[str] = None
    is_heuristic: Optional[bool] = None
    fallback_warning: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MeetingIntelligenceResponse(BaseModel):
    """Comprehensive meeting intelligence aggregation response."""
    meeting_id: str
    summary: Optional[MeetingSummaryResponse] = None
    topics: List[TopicResponse] = Field(default_factory=list)
    decisions: List[DecisionResponse] = Field(default_factory=list)
    action_items: List[ActionItemResponse] = Field(default_factory=list)
    risks: List[RiskResponse] = Field(default_factory=list)
    unresolved_questions: List[UnresolvedQuestionResponse] = Field(default_factory=list)
    provider_used: Optional[str] = None
    is_heuristic: Optional[bool] = None
    fallback_warning: Optional[str] = None
    has_intelligence: bool = False

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def compute_has_intelligence(self) -> "MeetingIntelligenceResponse":
        self.has_intelligence = bool(
            self.summary is not None
            or self.topics
            or self.decisions
            or self.action_items
            or self.risks
            or self.unresolved_questions
        )
        return self


# --- Request & Action Schemas ---

class DecisionStatusUpdate(BaseModel):
    """Payload to update the lifecycle status or lineage of a decision."""
    status: DecisionStatus = Field(..., description="Target status (ACTIVE, SUPERSEDED, REVOKED)")
    supersedes_decision_id: Optional[str] = Field(None, description="Optional UUID of earlier decision superseded")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return DecisionStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid decision status '{v}'. Allowed values: {[s.value for s in DecisionStatus]}"
                )
        return v


class DecisionSupersedeCreate(BaseModel):
    """Payload to create a new decision that explicitly supersedes an existing target decision."""
    meeting_id: str = Field(..., min_length=1, description="UUID of meeting where new decision was reached")
    description: str = Field(..., min_length=1, description="Statement of the new superseding decision")
    rationale_or_context: Optional[str] = Field(None, description="Why the earlier decision was superseded or context")
    decided_by: Optional[str] = Field(None, max_length=255, description="Decision maker or committee")
    evidence_segment_id: Optional[str] = Field(None, description="Foreign key to transcript segment supporting new decision")
    timestamp_seconds: Optional[float] = Field(None, ge=0.0, description="Timeline timestamp of new decision in seconds")


class DecisionLinkSupersededRequest(BaseModel):
    """Payload to link an existing decision as superseded by the target decision."""
    superseded_decision_id: str = Field(..., min_length=1, description="UUID of earlier decision to mark as superseded")


class DecisionRevokeRequest(BaseModel):
    """Payload to revoke an active decision with an audited reason."""
    reason: str = Field(..., min_length=1, max_length=1000, description="Mandatory rationale for revoking the decision")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("Revocation reason cannot be empty or whitespace-only.")
        return v_clean


class ActionItemStatusUpdate(BaseModel):
    """Payload to update the lifecycle status of an action item."""
    status: ActionItemStatus = Field(..., description="Target status (PENDING, IN_PROGRESS, COMPLETED, CANCELLED)")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return ActionItemStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid action item status '{v}'. Allowed values: {[s.value for s in ActionItemStatus]}"
                )
        return v


class CommitmentReviewUpdate(BaseModel):
    """Payload to record cross-meeting review/status update of a commitment."""
    reviewing_meeting_id: str = Field(..., min_length=1, description="Meeting UUID where commitment was reviewed")
    evidence_segment_id: Optional[str] = Field(None, description="Transcript segment providing evidence for the review")
    status: Optional[ActionItemStatus] = Field(None, description="Updated status if changed")
    review_notes: Optional[str] = Field(None, description="Context or notes from the review discussion")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if v is not None and isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return ActionItemStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid action item status '{v}'. Allowed values: {[s.value for s in ActionItemStatus]}"
                )
        return v


class QuestionResolutionUpdate(BaseModel):
    """Payload to resolve an unresolved question."""
    resolving_meeting_id: str = Field(..., min_length=1, description="Meeting UUID where question was resolved")
    resolving_decision_id: Optional[str] = Field(None, description="Decision UUID that answers/resolves this question")
    resolution_notes: Optional[str] = Field(None, description="Explanation or summary of the resolution")
    status: QuestionStatus = Field(QuestionStatus.RESOLVED, description="Updated question status (defaults to RESOLVED)")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            try:
                return QuestionStatus(v_upper)
            except ValueError:
                raise ValueError(
                    f"Invalid question status '{v}'. Allowed values: {[s.value for s in QuestionStatus]}"
                )
        return v


class ExtractIntelligenceRequest(BaseModel):
    """Request payload to trigger asynchronous or synchronous intelligence extraction."""
    source_transcript_id: Optional[str] = Field(
        None,
        description="Optional source transcript ID. Defaults to latest privacy-safe transcript (CLEANED_REDACTED or REDACTED)."
    )
    output_language: Optional[str] = Field(
        "en",
        description="Language code for summary and action synthesis (default 'en')."
    )
    provider_override: Optional[str] = Field(
        None,
        description="Optional provider override ('gemini', 'deterministic', 'ollama'). Defaults to system configured provider."
    )


class ExtractionJobPayload(BaseModel):
    """Schema for validating background extraction job payloads stored in ProcessingJob.payload."""
    meeting_id: str = Field(..., min_length=1, description="Target meeting UUID (must match job.meeting_id)")
    source_transcript_id: Optional[str] = Field(
        None,
        description="Optional source transcript ID. If omitted, resolves the latest privacy-safe transcript."
    )
    output_language: Optional[str] = Field(
        "en",
        min_length=2,
        max_length=20,
        description="Target output language code (e.g., 'en', 'hi', 'gu')."
    )
    provider_override: Optional[str] = Field(
        None,
        description="Optional provider override ('gemini', 'deterministic', 'ollama')."
    )
    organization_id: Optional[str] = Field(
        None,
        description="Optional org ID for consistency validation against the meeting."
    )

    @field_validator("provider_override")
    @classmethod
    def validate_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_norm = v.strip().lower()
            if v_norm not in {"ollama", "deterministic", "disabled", "gemini"}:
                raise ValueError(f"Invalid provider_override '{v}'. Allowed values: 'gemini', 'ollama', 'deterministic', 'disabled'.")
            return v_norm
        return v


class ExtractIntelligenceJobResponse(BaseModel):
    """Response returned upon successfully enqueuing an extraction job."""
    job_id: str
    meeting_id: str
    job_type: str = "EXTRACTION"
    status: str
    message: str

    model_config = ConfigDict(from_attributes=True)
