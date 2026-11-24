import enum
from typing import Optional, List
from sqlalchemy import Column, String, Boolean, Text, ForeignKey, Float, BigInteger, DateTime, Integer
from sqlalchemy.orm import relationship
from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin


class InputSourceType(str, enum.Enum):
    """Supported input source types for meetings."""
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    YOUTUBE = "YOUTUBE"
    LIVE_STREAM = "LIVE_STREAM"


class MeetingStatus(str, enum.Enum):
    """Lifecycle statuses for a meeting."""
    SCHEDULED = "SCHEDULED"
    RECORDING = "RECORDING"
    INGESTING = "INGESTING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobType(str, enum.Enum):
    """Background processing job types."""
    INGESTION = "INGESTION"
    TRANSCRIPTION = "TRANSCRIPTION"
    TRANSCRIPT_PROCESSING = "TRANSCRIPT_PROCESSING"
    DIARIZATION = "DIARIZATION"
    SUMMARIZATION = "SUMMARIZATION"
    EXTRACTION = "EXTRACTION"
    EMBEDDING = "EMBEDDING"


class JobStatus(str, enum.Enum):
    """Status of background processing jobs."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Meeting(UUIDMixin, TimestampMixin, Base):
    """Primary meeting record within an organization."""
    __tablename__ = "meetings"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    input_source_type = Column(
        String(50),
        default=InputSourceType.AUDIO.value,
        nullable=False
    )
    source_url_or_reference = Column(String(1024), nullable=True)
    status = Column(
        String(50),
        default=MeetingStatus.SCHEDULED.value,
        nullable=False,
        index=True
    )
    scheduled_start_time = Column(DateTime(timezone=True), nullable=True)
    actual_start_time = Column(DateTime(timezone=True), nullable=True)
    actual_end_time = Column(DateTime(timezone=True), nullable=True)
    category = Column(String(50), nullable=True, index=True)
    project_id = Column(
        String(36),
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    created_by_user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Relationships
    organization = relationship("Organization", back_populates="meetings")
    project = relationship("Project", back_populates="meetings")

    @property
    def project_name(self) -> Optional[str]:
        return self.project.name if self.project else None

    @property
    def transcript_types(self) -> List[str]:
        if not self.transcripts:
            return []
        return [t.transcript_type for t in self.transcripts]
    created_by_user = relationship(
        "User",
        back_populates="created_meetings",
        foreign_keys=[created_by_user_id]
    )
    participants = relationship(
        "MeetingParticipant",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    media_assets = relationship(
        "MediaAsset",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    processing_jobs = relationship(
        "ProcessingJob",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    speakers = relationship(
        "Speaker",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    transcripts = relationship(
        "Transcript",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    summaries = relationship(
        "MeetingSummary",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    topics = relationship(
        "Topic",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    decisions = relationship(
        "Decision",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    action_items = relationship(
        "ActionItem",
        back_populates="meeting",
        cascade="all, delete-orphan",
        foreign_keys="ActionItem.meeting_id",
    )
    risks = relationship(
        "Risk",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    unresolved_questions = relationship(
        "UnresolvedQuestion",
        back_populates="meeting",
        cascade="all, delete-orphan",
        foreign_keys="UnresolvedQuestion.meeting_id",
    )
    source_references = relationship(
        "SourceReference",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )


class MeetingParticipant(UUIDMixin, TimestampMixin, Base):
    """Attendee or participant in a meeting (registered user or guest)."""
    __tablename__ = "meeting_participants"

    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    display_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    role = Column(String(50), nullable=True)

    # Relationships
    meeting = relationship("Meeting", back_populates="participants")
    user = relationship("User")


class MediaAsset(UUIDMixin, TimestampMixin, Base):
    """Metadata record for audio/video media linked to a meeting.

    IMPORTANT: Never stores raw audio/video binary contents in SQLite.
    Stores storage metadata, path/URL references, and handles raw audio deletion flags.
    """
    __tablename__ = "media_assets"

    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    file_name = Column(String(255), nullable=False)
    storage_path = Column(String(1024), nullable=False)  # Local filesystem path or object key
    media_type = Column(String(50), nullable=False)  # AUDIO, VIDEO, YOUTUBE_STREAM
    mime_type = Column(String(100), nullable=False)  # e.g., audio/wav, video/mp4
    file_size_bytes = Column(BigInteger, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    checksum_sha256 = Column(String(64), nullable=True)
    is_raw_deleted = Column(Boolean, default=False, nullable=False)  # Privacy policy tracking

    # Relationships
    meeting = relationship("Meeting", back_populates="media_assets")


class ProcessingJob(UUIDMixin, TimestampMixin, Base):
    """Background processing job tracking (transcription, diarization, summarization)."""
    __tablename__ = "processing_jobs"

    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    job_type = Column(String(50), nullable=False)
    status = Column(
        String(50),
        default=JobStatus.PENDING.value,
        nullable=False,
        index=True
    )
    progress = Column(Float, default=0.0, nullable=False)  # 0.0 to 100.0 percent
    error_message = Column(Text, nullable=True)
    error_details = Column(Text, nullable=True)
    attempt_count = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=2, nullable=False)
    payload = Column(Text, nullable=True)  # JSON-serialized execution parameters
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    meeting = relationship("Meeting", back_populates="processing_jobs")
