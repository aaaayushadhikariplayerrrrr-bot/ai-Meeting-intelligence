import enum
from sqlalchemy import Column, String, Boolean, Text, ForeignKey, Float, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin


class TranscriptType(str, enum.Enum):
    """Transcript versions supported by the system."""
    ORIGINAL = "ORIGINAL"
    CLEANED = "CLEANED"
    REDACTED = "REDACTED"
    CLEANED_REDACTED = "CLEANED_REDACTED"
    TRANSLATED = "TRANSLATED"


class Speaker(UUIDMixin, TimestampMixin, Base):
    """Identified speaker in a meeting (diarization cluster or mapped user)."""
    __tablename__ = "speakers"

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
    speaker_label = Column(String(100), nullable=False)  # e.g., "SPEAKER_00", "SPEAKER_01"
    display_name = Column(String(255), nullable=True)   # Human-assigned or recognized name

    # Relationships
    meeting = relationship("Meeting", back_populates="speakers")
    user = relationship("User")
    segments = relationship("TranscriptSegment", back_populates="speaker")


class Transcript(UUIDMixin, TimestampMixin, Base):
    """Transcript container preserving versions (original vs cleaned/redacted).

    IMPORTANT: Multiple transcript versions (e.g. ORIGINAL, CLEANED_REDACTED,
    TRANSLATED) can coexist for a single meeting without overwriting the original.
    """
    __tablename__ = "transcripts"

    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    transcript_type = Column(
        String(50),
        default=TranscriptType.ORIGINAL.value,
        nullable=False,
        index=True
    )
    version = Column(Integer, default=1, nullable=False)
    language_detected = Column(String(50), nullable=True)  # Overall language detected
    is_primary = Column(Boolean, default=False, nullable=False)
    parent_transcript_id = Column(
        String(36),
        ForeignKey("transcripts.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Relationships
    meeting = relationship("Meeting", back_populates="transcripts")
    parent_transcript = relationship("Transcript", remote_side="Transcript.id", foreign_keys=[parent_transcript_id])
    segments = relationship(
        "TranscriptSegment",
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.sequence_order",
        foreign_keys="TranscriptSegment.transcript_id"
    )
    summaries = relationship("MeetingSummary", back_populates="transcript")

    __table_args__ = (
        UniqueConstraint("meeting_id", "transcript_type", "version", name="uq_meeting_transcript_type_version"),
    )


class TranscriptSegment(UUIDMixin, TimestampMixin, Base):
    """Individual utterance/timestamp segment within a transcript.

    Supports multilingual speech segments (Hindi 'hi', English 'en',
    Gujarati 'gu', Hinglish/code-switching 'hi-en', etc.). Each segment can have
    a distinct language code without assuming the entire meeting has only one language.
    """
    __tablename__ = "transcript_segments"

    transcript_id = Column(
        String(36),
        ForeignKey("transcripts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    speaker_id = Column(
        String(36),
        ForeignKey("speakers.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    sequence_order = Column(Integer, nullable=False, index=True)
    start_time = Column(Float, nullable=False, index=True)  # Seconds from audio start
    end_time = Column(Float, nullable=False)                # Seconds
    text = Column(Text, nullable=False)
    language_code = Column(
        String(20),
        default="en",
        nullable=False,
        index=True
    )  # 'hi', 'en', 'gu', 'hi-en', 'hinglish'
    confidence = Column(Float, nullable=True)               # ASR confidence 0.0 - 1.0
    is_redacted = Column(Boolean, default=False, nullable=False)
    source_segment_id = Column(
        String(36),
        ForeignKey("transcript_segments.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    redaction_metadata = Column(Text, nullable=True)  # JSON-encoded array of detected categories (e.g. '["EMAIL"]')

    def __init__(self, **kwargs):
        if "segment_index" in kwargs and "sequence_order" not in kwargs:
            kwargs["sequence_order"] = kwargs.pop("segment_index")
        if "language" in kwargs and "language_code" not in kwargs:
            kwargs["language_code"] = kwargs.pop("language")
        super().__init__(**kwargs)

    # Relationships
    transcript = relationship(
        "Transcript",
        back_populates="segments",
        foreign_keys=[transcript_id]
    )
    source_segment = relationship(
        "TranscriptSegment",
        remote_side="TranscriptSegment.id",
        foreign_keys=[source_segment_id]
    )
    speaker = relationship("Speaker", back_populates="segments")
    decisions = relationship("Decision", back_populates="evidence_segment")
    action_items = relationship(
        "ActionItem",
        back_populates="evidence_segment",
        foreign_keys="ActionItem.evidence_segment_id",
    )
    risks = relationship("Risk", back_populates="evidence_segment")
    unresolved_questions = relationship("UnresolvedQuestion", back_populates="evidence_segment")
    source_references = relationship("SourceReference", back_populates="transcript_segment")
