"""Models package exporting all domain entities for SQLAlchemy and Alembic."""
from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin, utc_now
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import (
    Meeting,
    MeetingParticipant,
    MediaAsset,
    ProcessingJob,
    InputSourceType,
    MeetingStatus,
    JobType,
    JobStatus,
)
from app.models.transcription import (
    Speaker,
    Transcript,
    TranscriptSegment,
    TranscriptType,
)
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
    ActionItemStatus,
    RiskSeverity,
    QuestionStatus,
)
from app.models.memory import (
    KnowledgeRecord,
    SourceReference,
    AuditLog,
    TranscriptChunkRecord,
    KnowledgeSourceType,
    ReferenceType,
)

__all__ = [
    "Base",
    "UUIDMixin",
    "TimestampMixin",
    "utc_now",
    # Identity
    "User",
    "Organization",
    "OrganizationMembership",
    "OrgRole",
    # Meeting
    "Meeting",
    "MeetingParticipant",
    "MediaAsset",
    "ProcessingJob",
    "InputSourceType",
    "MeetingStatus",
    "JobType",
    "JobStatus",
    # Transcription
    "Speaker",
    "Transcript",
    "TranscriptSegment",
    "TranscriptType",
    # Intelligence
    "MeetingSummary",
    "Topic",
    "Decision",
    "ActionItem",
    "Risk",
    "UnresolvedQuestion",
    "ActionItemStatus",
    "RiskSeverity",
    "QuestionStatus",
    # Memory & Evidence
    "KnowledgeRecord",
    "SourceReference",
    "AuditLog",
    "TranscriptChunkRecord",
    "KnowledgeSourceType",
    "ReferenceType",
]
