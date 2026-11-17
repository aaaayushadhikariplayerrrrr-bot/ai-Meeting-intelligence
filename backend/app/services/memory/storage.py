"""Persistent SQLite vector storage and FTS5 synchronization for Multilingual Semantic Chunks."""

import json
import uuid
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.base import utc_now
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptType
from app.services.memory.exceptions import (
    VectorStorageError,
    VectorValidationError,
    FTS5UnavailableError,
    PrivacyEnforcementError,
    TenantSecurityError,
)
from app.services.memory.types import ChunkEmbeddingResult, SemanticChunk

EXPECTED_DIMENSION: int = 384
FLOAT32_BYTES_PER_ELEMENT: int = 4
EXPECTED_BYTE_LENGTH: int = EXPECTED_DIMENSION * FLOAT32_BYTES_PER_ELEMENT  # 1536 bytes


def serialize_embedding(
    vector: Union[List[float], np.ndarray],
    expected_dim: int = EXPECTED_DIMENSION,
) -> bytes:
    """Validate and serialize a dense embedding vector to a raw float32 binary BLOB.

    Enforces:
    1. Exact dimension match (default 384 for multilingual-e5-small).
    2. Finite values (no NaN or Inf).
    3. Unit L2 normalization (within numerical tolerance 1e-3).
    """
    if vector is None:
        raise VectorValidationError("Vector cannot be None.")

    try:
        arr = np.asarray(vector, dtype=np.float32)
    except Exception as exc:
        raise VectorValidationError(f"Failed to parse vector as float32 array: {exc}") from exc

    if arr.ndim != 1:
        raise VectorValidationError(f"Expected 1D vector, got array with shape {arr.shape}.")

    if arr.shape[0] != expected_dim:
        raise VectorValidationError(
            f"Embedding vector dimension mismatch: expected {expected_dim}, got {arr.shape[0]}."
        )

    if not np.all(np.isfinite(arr)):
        raise VectorValidationError("Embedding vector contains non-finite values (NaN or Inf).")

    norm = float(np.linalg.norm(arr))
    if abs(norm - 1.0) > 1e-3:
        raise VectorValidationError(
            f"Embedding vector is not unit normalized: L2 norm is {norm:.6f}, expected 1.0."
        )

    return arr.tobytes()


def deserialize_embedding(
    blob: bytes,
    expected_dim: int = EXPECTED_DIMENSION,
) -> np.ndarray:
    """Validate and deserialize a raw binary BLOB back into a 1D float32 NumPy array."""
    if not isinstance(blob, (bytes, bytearray)):
        raise VectorValidationError(f"Expected bytes or bytearray, got {type(blob)}.")

    expected_len = expected_dim * FLOAT32_BYTES_PER_ELEMENT
    if len(blob) != expected_len:
        raise VectorValidationError(
            f"Binary embedding byte length mismatch: expected {expected_len} bytes, got {len(blob)} bytes."
        )

    arr = np.frombuffer(blob, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise VectorValidationError("Deserialized embedding vector contains non-finite values.")

    return arr


def check_fts5_available(db: Session) -> bool:
    """Check if SQLite FTS5 extension is available and functional in the database connection."""
    try:
        bind = db.get_bind()
        if bind.dialect.name != "sqlite":
            return False
        db.execute(text("CREATE VIRTUAL TABLE IF NOT EXISTS temp._fts5_check USING fts5(x)"))
        db.execute(text("DROP TABLE IF EXISTS temp._fts5_check"))
        return True
    except Exception:
        return False


def ensure_fts5_ready(db: Session) -> None:
    """Validate that FTS5 is available and raise FTS5UnavailableError if not."""
    if not check_fts5_available(db):
        raise FTS5UnavailableError(
            "SQLite FTS5 extension is unavailable or disabled in this database environment. "
            "Hybrid retrieval and keyword search require SQLite with FTS5 enabled."
        )


class ChunkStorageManager:
    """Manages persistence, validation, and lifecycle synchronization of semantic chunks,

    raw float32 vector BLOBs, and SQLite FTS5 full-text index records.
    Strictly isolates all storage and deletion operations by tenant organization_id.
    """

    def __init__(self, expected_dim: int = EXPECTED_DIMENSION):
        self.expected_dim = expected_dim

    def _verify_privacy_safe(
        self,
        db: Session,
        transcript_id: str,
        organization_id: str,
        transcript_type_hint: Optional[str] = None,
    ) -> None:
        """Enforce strict privacy: never index or store ORIGINAL or CLEANED unredacted transcripts."""
        if transcript_type_hint in (TranscriptType.ORIGINAL.value, TranscriptType.CLEANED.value):
            raise PrivacyEnforcementError(
                f"Cannot index unredacted transcript type '{transcript_type_hint}'. "
                "Only REDACTED or CLEANED_REDACTED transcripts may be stored or indexed."
            )

        transcript = (
            db.query(Transcript)
            .filter(Transcript.id == transcript_id)
            .first()
        )
        if transcript:
            if transcript.meeting and transcript.meeting.organization_id != organization_id:
                raise TenantSecurityError(
                    f"Transcript '{transcript_id}' belongs to organization "
                    f"'{transcript.meeting.organization_id}', not '{organization_id}'."
                )
            if transcript.transcript_type in (
                TranscriptType.ORIGINAL.value,
                TranscriptType.CLEANED.value,
            ):
                raise PrivacyEnforcementError(
                    f"Cannot index unredacted transcript type '{transcript.transcript_type}'. "
                    "Only REDACTED or CLEANED_REDACTED transcripts may be stored or indexed."
                )

    def store_chunks(
        self,
        db: Session,
        chunks_with_embeddings: List[ChunkEmbeddingResult],
        organization_id: str,
        transcript: Optional[Transcript] = None,
    ) -> List[TranscriptChunkRecord]:
        """Store semantic chunks and their normalized embedding vectors into the database

        while synchronizing the SQLite FTS5 index.

        Strictly enforces:
        1. Tenant organization boundary on all chunks.
        2. Rejection of unredacted transcripts (ORIGINAL / CLEANED).
        3. Vector validation (dimension, finite, unit L2 norm).
        4. Atomic transaction synchronization between relational chunks and FTS5 records.
        """
        if not organization_id or not organization_id.strip():
            raise TenantSecurityError("A valid organization_id is required to store chunks.")

        if not chunks_with_embeddings:
            return []

        ensure_fts5_ready(db)

        # Validate transcript if provided
        if transcript is not None:
            if transcript.meeting and transcript.meeting.organization_id != organization_id:
                raise TenantSecurityError(
                    f"Transcript '{transcript.id}' does not belong to organization '{organization_id}'."
                )
            if transcript.transcript_type in (
                TranscriptType.ORIGINAL.value,
                TranscriptType.CLEANED.value,
            ):
                raise PrivacyEnforcementError(
                    f"Cannot index unredacted transcript type '{transcript.transcript_type}'. "
                    "Only REDACTED or CLEANED_REDACTED transcripts may be stored or indexed."
                )

        # Pre-validate all items before persisting any record
        prepared_data = []
        for item in chunks_with_embeddings:
            chunk = item.chunk
            if chunk.organization_id != organization_id:
                raise TenantSecurityError(
                    f"Chunk organization_id '{chunk.organization_id}' does not match "
                    f"target organization_id '{organization_id}'."
                )

            type_hint = chunk.metadata.get("transcript_type")
            self._verify_privacy_safe(
                db=db,
                transcript_id=chunk.transcript_id,
                organization_id=organization_id,
                transcript_type_hint=type_hint,
            )

            # Validate and serialize embedding vector
            blob = serialize_embedding(item.embedding, expected_dim=self.expected_dim)
            prepared_data.append((chunk, item, blob))

        persisted_records: List[TranscriptChunkRecord] = []

        for chunk, item, blob in prepared_data:
            chunk_id = chunk.metadata.get("chunk_id") or str(uuid.uuid4())

            # Check if chunk record already exists
            existing_record = (
                db.query(TranscriptChunkRecord)
                .filter(
                    TranscriptChunkRecord.id == chunk_id,
                    TranscriptChunkRecord.organization_id == organization_id,
                )
                .first()
            )

            if existing_record:
                # Update existing record
                existing_record.text = chunk.text
                existing_record.start_time = chunk.start_time
                existing_record.end_time = chunk.end_time
                existing_record.segment_ids = json.dumps(chunk.segment_ids)
                existing_record.language_codes = json.dumps(chunk.language_codes)
                existing_record.dominant_language = chunk.dominant_language
                existing_record.token_count = chunk.token_count
                existing_record.is_split_utterance = chunk.is_split_utterance
                existing_record.extra_metadata = json.dumps(chunk.metadata)
                existing_record.model_name = item.model_name
                existing_record.dimension = item.dimension
                existing_record.embedding = blob
                existing_record.updated_at = utc_now()

                # Synchronize FTS5 index
                db.execute(
                    text("DELETE FROM transcript_chunks_fts WHERE chunk_id = :chunk_id"),
                    {"chunk_id": chunk_id},
                )
                db.execute(
                    text("INSERT INTO transcript_chunks_fts (chunk_id, text) VALUES (:chunk_id, :text)"),
                    {"chunk_id": chunk_id, "text": chunk.text},
                )
                persisted_records.append(existing_record)
            else:
                # Insert new record
                new_record = TranscriptChunkRecord(
                    id=chunk_id,
                    organization_id=organization_id,
                    meeting_id=chunk.meeting_id,
                    transcript_id=chunk.transcript_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    segment_ids=json.dumps(chunk.segment_ids),
                    language_codes=json.dumps(chunk.language_codes),
                    dominant_language=chunk.dominant_language,
                    token_count=chunk.token_count,
                    is_split_utterance=chunk.is_split_utterance,
                    extra_metadata=json.dumps(chunk.metadata),
                    model_name=item.model_name,
                    dimension=item.dimension,
                    embedding=blob,
                )
                db.add(new_record)
                db.flush()

                # Synchronize FTS5 index
                db.execute(
                    text("INSERT INTO transcript_chunks_fts (chunk_id, text) VALUES (:chunk_id, :text)"),
                    {"chunk_id": chunk_id, "text": chunk.text},
                )
                persisted_records.append(new_record)

        return persisted_records

    def update_chunk(
        self,
        db: Session,
        chunk_id: str,
        organization_id: str,
        text_content: Optional[str] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptChunkRecord:
        """Update an existing chunk record and synchronize FTS5. Strictly respects organization ownership."""
        ensure_fts5_ready(db)

        record = (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.id == chunk_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .first()
        )
        if not record:
            raise VectorStorageError(
                f"Chunk '{chunk_id}' not found for organization '{organization_id}'."
            )

        if text_content is not None:
            record.text = text_content
            db.execute(
                text("UPDATE transcript_chunks_fts SET text = :text WHERE chunk_id = :chunk_id"),
                {"chunk_id": chunk_id, "text": text_content},
            )

        if embedding is not None:
            record.embedding = serialize_embedding(embedding, expected_dim=self.expected_dim)

        if extra_metadata is not None:
            record.extra_metadata = json.dumps(extra_metadata)

        record.updated_at = utc_now()
        db.flush()
        return record

    def delete_chunks_by_transcript(
        self,
        db: Session,
        transcript_id: str,
        organization_id: str,
    ) -> int:
        """Delete all chunks and FTS5 entries belonging to a transcript within an organization."""
        ensure_fts5_ready(db)

        records = (
            db.query(TranscriptChunkRecord.id)
            .filter(
                TranscriptChunkRecord.transcript_id == transcript_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .all()
        )
        if not records:
            return 0

        chunk_ids = [r[0] for r in records]
        for cid in chunk_ids:
            db.execute(
                text("DELETE FROM transcript_chunks_fts WHERE chunk_id = :cid"),
                {"cid": cid},
            )

        deleted = (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.id.in_(chunk_ids),
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted

    def delete_chunks_by_meeting(
        self,
        db: Session,
        meeting_id: str,
        organization_id: str,
    ) -> int:
        """Delete all chunks and FTS5 entries belonging to a meeting within an organization."""
        ensure_fts5_ready(db)

        records = (
            db.query(TranscriptChunkRecord.id)
            .filter(
                TranscriptChunkRecord.meeting_id == meeting_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .all()
        )
        if not records:
            return 0

        chunk_ids = [r[0] for r in records]
        for cid in chunk_ids:
            db.execute(
                text("DELETE FROM transcript_chunks_fts WHERE chunk_id = :cid"),
                {"cid": cid},
            )

        deleted = (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.id.in_(chunk_ids),
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted

    def delete_chunks_by_organization(
        self,
        db: Session,
        organization_id: str,
    ) -> int:
        """Purge all chunks and FTS5 entries belonging to an organization."""
        ensure_fts5_ready(db)

        records = (
            db.query(TranscriptChunkRecord.id)
            .filter(TranscriptChunkRecord.organization_id == organization_id)
            .all()
        )
        if not records:
            return 0

        chunk_ids = [r[0] for r in records]
        for cid in chunk_ids:
            db.execute(
                text("DELETE FROM transcript_chunks_fts WHERE chunk_id = :cid"),
                {"cid": cid},
            )

        deleted = (
            db.query(TranscriptChunkRecord)
            .filter(TranscriptChunkRecord.organization_id == organization_id)
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted

    def get_chunks_by_transcript(
        self,
        db: Session,
        transcript_id: str,
        organization_id: str,
    ) -> List[TranscriptChunkRecord]:
        """Retrieve stored chunks for a transcript within an organization ordered by chunk_index."""
        return (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.transcript_id == transcript_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .order_by(TranscriptChunkRecord.chunk_index.asc())
            .all()
        )

    def get_chunks_by_meeting(
        self,
        db: Session,
        meeting_id: str,
        organization_id: str,
    ) -> List[TranscriptChunkRecord]:
        """Retrieve stored chunks for a meeting within an organization ordered by start_time."""
        return (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.meeting_id == meeting_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .order_by(TranscriptChunkRecord.start_time.asc())
            .all()
        )

    def get_chunk_by_id(
        self,
        db: Session,
        chunk_id: str,
        organization_id: str,
    ) -> Optional[TranscriptChunkRecord]:
        """Retrieve a specific chunk record by ID enforcing organization boundary."""
        return (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.id == chunk_id,
                TranscriptChunkRecord.organization_id == organization_id,
            )
            .first()
        )
