import enum
import json
from typing import List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Text,
    ForeignKey,
    Float,
    DateTime,
    Integer,
    LargeBinary,
    Index,
    event,
    text,
)
from sqlalchemy.orm import relationship
from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin, utc_now


class KnowledgeSourceType(str, enum.Enum):
    """Categorization of knowledge in the organizational memory."""
    INTERNAL_MEETING = "INTERNAL_MEETING"
    EXTERNAL_REFERENCE = "EXTERNAL_REFERENCE"


class ReferenceType(str, enum.Enum):
    """Type of source reference link for evidence-grounded attribution."""
    MEETING_SEGMENT = "MEETING_SEGMENT"
    EXTERNAL_DOCUMENT = "EXTERNAL_DOCUMENT"
    WEB_URL = "WEB_URL"


class KnowledgeRecord(UUIDMixin, TimestampMixin, Base):
    """Cross-meeting organizational memory and external knowledge repository.

    IMPORTANT: Logically separates internal meeting knowledge from external reference
    knowledge (via knowledge_type) to support intelligent RAG source routing.
    """
    __tablename__ = "knowledge_records"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    knowledge_type = Column(
        String(50),
        default=KnowledgeSourceType.INTERNAL_MEETING.value,
        nullable=False,
        index=True
    )
    category = Column(String(100), nullable=True)  # e.g., "DECISION", "POLICY", "SYSTEM_DESIGN"
    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    organization = relationship("Organization", back_populates="knowledge_records")
    meeting = relationship("Meeting")
    source_references = relationship(
        "SourceReference",
        back_populates="knowledge_record",
        cascade="all, delete-orphan"
    )


class SourceReference(UUIDMixin, TimestampMixin, Base):
    """Evidence grounding reference linking knowledge to meetings, segments, or external URLs.

    Enables dual-source RAG to cite specific meeting timestamps or external resources.
    """
    __tablename__ = "source_references"

    knowledge_record_id = Column(
        String(36),
        ForeignKey("knowledge_records.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    transcript_segment_id = Column(
        String(36),
        ForeignKey("transcript_segments.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    reference_type = Column(
        String(50),
        default=ReferenceType.MEETING_SEGMENT.value,
        nullable=False
    )
    external_url = Column(String(1024), nullable=True)
    external_title = Column(String(255), nullable=True)
    timestamp_start = Column(Float, nullable=True)
    timestamp_end = Column(Float, nullable=True)
    text_snippet = Column(Text, nullable=True)

    # Relationships
    knowledge_record = relationship("KnowledgeRecord", back_populates="source_references")
    meeting = relationship("Meeting", back_populates="source_references")
    transcript_segment = relationship("TranscriptSegment", back_populates="source_references")


class AuditLog(UUIDMixin, Base):
    """Immutable audit trail for compliance, privacy actions, and data access."""
    __tablename__ = "audit_logs"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    action = Column(String(100), nullable=False, index=True)  # e.g., "VIEW", "REDACT_PII", "DELETE_RAW_AUDIO"
    entity_type = Column(String(100), nullable=False)        # e.g., "Meeting", "Transcript", "MediaAsset"
    entity_id = Column(String(36), nullable=True, index=True)
    details = Column(Text, nullable=True)                    # JSON metadata or details
    ip_address = Column(String(45), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True
    )

    # Relationships
    organization = relationship("Organization", back_populates="audit_logs")
    user = relationship("User")


@event.listens_for(AuditLog, "before_update")
def _audit_log_prevent_update(mapper, connection, target):
    """Enforce append-only semantics: AuditLog rows cannot be modified."""
    raise RuntimeError(
        "AuditLog records are strictly append-only and cannot be updated or modified."
    )


@event.listens_for(AuditLog, "before_delete")
def _audit_log_prevent_delete(mapper, connection, target):
    """Enforce append-only semantics: AuditLog rows cannot be deleted."""
    raise RuntimeError(
        "AuditLog records are strictly append-only and cannot be deleted."
    )


class TranscriptChunkRecord(UUIDMixin, TimestampMixin, Base):
    """Persistent storage for semantic transcript chunks and normalized dense embeddings.

    Retains full evidence grounding metadata (segment IDs, timestamps, language information)
    and strictly isolates storage by tenant organization_id.
    """
    __tablename__ = "transcript_chunks"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    meeting_id = Column(
        String(36),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transcript_id = Column(
        String(36),
        ForeignKey("transcripts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    segment_ids = Column(Text, nullable=False)  # JSON-encoded List[str] of transcript_segment IDs
    language_codes = Column(Text, nullable=False)  # JSON-encoded List[str]
    dominant_language = Column(String(20), default="en", nullable=False)
    token_count = Column(Integer, nullable=False)
    is_split_utterance = Column(Boolean, default=False, nullable=False)
    extra_metadata = Column(Text, nullable=True)  # JSON-encoded Dict[str, Any]
    model_name = Column(String(100), default="intfloat/multilingual-e5-small", nullable=False)
    dimension = Column(Integer, default=384, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # float32 binary blob: exactly 384 * 4 = 1536 bytes

    __table_args__ = (
        Index("ix_transcript_chunks_org_meeting", "organization_id", "meeting_id"),
        Index("ix_transcript_chunks_org_transcript", "organization_id", "transcript_id"),
    )

    # Relationships
    organization = relationship("Organization")
    meeting = relationship("Meeting")
    transcript = relationship("Transcript")

    @property
    def segment_ids_list(self) -> List[str]:
        if not self.segment_ids:
            return []
        try:
            return json.loads(self.segment_ids)
        except Exception:
            return []

    @property
    def language_codes_list(self) -> List[str]:
        if not self.language_codes:
            return []
        try:
            return json.loads(self.language_codes)
        except Exception:
            return []

    @property
    def extra_metadata_dict(self) -> Dict[str, Any]:
        if not self.extra_metadata:
            return {}
        try:
            return json.loads(self.extra_metadata)
        except Exception:
            return {}


@event.listens_for(TranscriptChunkRecord.__table__, "after_create")
def _create_transcript_chunks_fts_table(target, connection, **kw):
    """Automatically create the SQLite FTS5 virtual table and delete trigger during Base.metadata.create_all."""
    if connection.dialect.name == "sqlite":
        try:
            connection.execute(text(
                "CREATE VIRTUAL TABLE IF NOT EXISTS transcript_chunks_fts USING fts5("
                "chunk_id UNINDEXED, text, tokenize='unicode61')"
            ))
            connection.execute(text(
                "CREATE TRIGGER IF NOT EXISTS trg_transcript_chunks_fts_delete "
                "AFTER DELETE ON transcript_chunks "
                "BEGIN "
                "    DELETE FROM transcript_chunks_fts WHERE chunk_id = old.id; "
                "END;"
            ))
        except Exception:
            pass


@event.listens_for(TranscriptChunkRecord.__table__, "before_drop")
def _drop_transcript_chunks_fts_table(target, connection, **kw):
    """Automatically drop the SQLite FTS5 virtual table and trigger during Base.metadata.drop_all."""
    if connection.dialect.name == "sqlite":
        try:
            connection.execute(text("DROP TRIGGER IF EXISTS trg_transcript_chunks_fts_delete"))
            connection.execute(text("DROP TABLE IF EXISTS transcript_chunks_fts"))
        except Exception:
            pass
