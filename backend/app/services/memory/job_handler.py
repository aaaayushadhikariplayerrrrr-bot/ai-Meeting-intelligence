"""Background Job Handler for Automatic Privacy-Safe Memory Indexing."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Any, List
from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.models.meeting import Meeting, ProcessingJob, JobStatus, JobType
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.schemas.memory import IndexingJobPayload
from app.services.job_manager import create_processing_job
from app.services.meeting_lifecycle import reconcile_meeting_status
from app.services.memory.chunking import SemanticTranscriptChunker
from app.services.memory.embedding import (
    MultilingualE5EmbeddingService,
    get_embedding_service,
)
from app.services.memory.exceptions import (
    MemoryError,
    ModelNotCachedError,
    EmbeddingModelLoadError,
    PrivacyEnforcementError,
    TenantSecurityError,
    VectorStorageError,
    VectorValidationError,
)
from app.services.memory.storage import ChunkStorageManager, EXPECTED_DIMENSION
from app.services.memory.types import ChunkEmbeddingResult

logger = logging.getLogger(__name__)


def schedule_indexing_job(
    db: Session,
    meeting_id: str,
    organization_id: str,
    transcript_id: Optional[str] = None,
    force_reindex: bool = False,
) -> ProcessingJob:
    """Schedule a durable background indexing job using the existing ProcessingJob model.

    Guarantees:
    1. Tenant organization and meeting ownership validation.
    2. Rejection of unredacted transcripts (ORIGINAL / CLEANED).
    3. Duplicate active job prevention (reuses existing PENDING or IN_PROGRESS job).
    4. Never logs transcript text or PII.
    """
    if not meeting_id or not meeting_id.strip():
        raise ValueError("meeting_id is required to schedule an indexing job.")
    if not organization_id or not organization_id.strip():
        raise TenantSecurityError("organization_id is required to schedule an indexing job.")

    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == organization_id)
        .first()
    )
    if not meeting:
        raise TenantSecurityError(f"Meeting '{meeting_id}' not found in organization '{organization_id}'.")

    # If transcript_id is specified, strictly validate it is privacy-safe
    if transcript_id:
        transcript = (
            db.query(Transcript)
            .filter(Transcript.id == transcript_id, Transcript.meeting_id == meeting_id)
            .first()
        )
        if not transcript:
            raise ValueError(f"Transcript '{transcript_id}' not found for meeting '{meeting_id}'.")
        if transcript.transcript_type in (
            TranscriptType.ORIGINAL.value,
            TranscriptType.CLEANED.value,
        ):
            raise PrivacyEnforcementError(
                f"Cannot index unredacted transcript type '{transcript.transcript_type}'. "
                "Only REDACTED or CLEANED_REDACTED transcripts may be indexed."
            )

    # Duplicate Prevention: Check for existing active (PENDING or IN_PROGRESS) indexing jobs
    active_jobs = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
            ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value]),
        )
        .all()
    )
    for active_job in active_jobs:
        if active_job.payload:
            try:
                active_payload = json.loads(active_job.payload)
                active_t_id = active_payload.get("transcript_id")
                # If target transcript matches (or both select default), reuse the existing job
                if active_t_id == transcript_id:
                    logger.info(
                        f"Reusing existing active indexing job '{active_job.id}' "
                        f"(status: {active_job.status}) for meeting '{meeting_id}'."
                    )
                    return active_job
            except Exception:
                pass

    # Create new durable job payload
    payload = IndexingJobPayload(
        meeting_id=meeting_id,
        organization_id=organization_id,
        transcript_id=transcript_id,
        force_reindex=force_reindex,
    )

    job = create_processing_job(
        db=db,
        meeting_id=meeting_id,
        job_type=JobType.EMBEDDING.value,
        initial_status=JobStatus.PENDING.value,
        payload=payload.model_dump(),
        max_attempts=2,
    )
    logger.info(f"Scheduled memory indexing job '{job.id}' for meeting '{meeting_id}'.")
    return job


def process_indexing_job_sync(
    job_id: str,
    embedding_service: Optional[MultilingualE5EmbeddingService] = None,
    chunker: Optional[SemanticTranscriptChunker] = None,
    storage_manager: Optional[ChunkStorageManager] = None,
) -> None:
    """Synchronous background handler executing an organizational memory indexing job.

    Designed to run on worker threads without blocking asyncio loops.
    """
    db: Session = SessionLocal()
    try:
        job = db.get(ProcessingJob, job_id)
        if not job:
            logger.error(f"ProcessingJob '{job_id}' not found.")
            return

        if job.status == JobStatus.COMPLETED.value:
            logger.info(f"ProcessingJob '{job_id}' is already completed; skipping execution.")
            return

        # 1. Transition to IN_PROGRESS
        job.status = JobStatus.IN_PROGRESS.value
        job.progress = 5.0
        now = datetime.now(timezone.utc)
        if not job.started_at:
            job.started_at = now
        db.commit()

        # 2. Parse & Validate Payload
        if not job.payload:
            job.status = JobStatus.FAILED.value
            job.error_message = "Malformed indexing job: payload is missing or empty."
            job.error_details = "MissingPayloadError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        try:
            raw_payload = json.loads(job.payload)
            if not isinstance(raw_payload, dict):
                raise ValueError("Payload must be a JSON object.")
            payload_dto = IndexingJobPayload.model_validate(raw_payload)
        except Exception as pe:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Malformed indexing job payload: {str(pe)}"
            job.error_details = "InvalidPayloadError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 3. Validate Meeting and Tenant Ownership
        if payload_dto.meeting_id != job.meeting_id:
            job.status = JobStatus.FAILED.value
            job.error_message = (
                f"Payload meeting_id '{payload_dto.meeting_id}' does not match job meeting_id '{job.meeting_id}'."
            )
            job.error_details = "MeetingMismatchError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        meeting = db.get(Meeting, job.meeting_id)
        if not meeting:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Meeting '{job.meeting_id}' not found."
            job.error_details = "MeetingNotFoundError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        if meeting.organization_id != payload_dto.organization_id:
            job.status = JobStatus.FAILED.value
            job.error_message = (
                f"Meeting '{meeting.id}' belongs to organization '{meeting.organization_id}', "
                f"not claimed organization '{payload_dto.organization_id}'."
            )
            job.error_details = "TenantSecurityError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 4. Resolve Target Privacy-Safe Transcript
        target_transcript: Optional[Transcript] = None
        if payload_dto.transcript_id:
            target_transcript = (
                db.query(Transcript)
                .filter(
                    Transcript.id == payload_dto.transcript_id,
                    Transcript.meeting_id == meeting.id,
                )
                .first()
            )
            if not target_transcript:
                job.status = JobStatus.FAILED.value
                job.error_message = f"Transcript '{payload_dto.transcript_id}' not found for meeting '{meeting.id}'."
                job.error_details = "TranscriptNotFoundError"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return

            if target_transcript.transcript_type in (
                TranscriptType.ORIGINAL.value,
                TranscriptType.CLEANED.value,
            ):
                job.status = JobStatus.FAILED.value
                job.error_message = (
                    f"Cannot index unredacted transcript type '{target_transcript.transcript_type}'. "
                    "Only REDACTED or CLEANED_REDACTED transcripts may be indexed."
                )
                job.error_details = "PrivacyEnforcementError"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return
        else:
            # Automatic Selection: Prefer CLEANED_REDACTED (latest), else REDACTED (latest)
            privacy_safe_transcripts = (
                db.query(Transcript)
                .filter(
                    Transcript.meeting_id == meeting.id,
                    Transcript.transcript_type.in_([
                        TranscriptType.CLEANED_REDACTED.value,
                        TranscriptType.REDACTED.value,
                    ]),
                )
                .all()
            )
            if not privacy_safe_transcripts:
                job.status = JobStatus.FAILED.value
                job.error_message = (
                    f"No privacy-safe (redacted) transcript available for meeting '{meeting.id}'. "
                    "Generate a REDACTED or CLEANED_REDACTED version first."
                )
                job.error_details = "SafeTranscriptRequiredError"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return

            priority_map = {
                TranscriptType.CLEANED_REDACTED.value: 0,
                TranscriptType.REDACTED.value: 1,
            }
            target_transcript = sorted(
                privacy_safe_transcripts,
                key=lambda t: (priority_map.get(t.transcript_type, 99), -t.version),
            )[0]

        # 5. Idempotency Check: Avoid redundant re-embedding if already indexed
        configured_service = embedding_service or get_embedding_service()
        model_name = getattr(configured_service, "model_name", "intfloat/multilingual-e5-small")
        dimension = getattr(configured_service, "dimension", EXPECTED_DIMENSION)

        existing_chunks_count = (
            db.query(TranscriptChunkRecord)
            .filter(
                TranscriptChunkRecord.transcript_id == target_transcript.id,
                TranscriptChunkRecord.organization_id == meeting.organization_id,
                TranscriptChunkRecord.model_name == model_name,
            )
            .count()
        )
        if existing_chunks_count > 0 and not payload_dto.force_reindex:
            logger.info(
                f"Transcript '{target_transcript.id}' is already indexed with {existing_chunks_count} chunks. "
                "Skipping re-embedding (force_reindex=False)."
            )
            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.error_message = None
            job.error_details = None
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 6. Semantic Chunking
        job.progress = 20.0
        db.commit()

        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == target_transcript.id)
            .order_by(TranscriptSegment.sequence_order.asc())
            .all()
        )
        if not segments:
            job.status = JobStatus.FAILED.value
            job.error_message = f"Transcript '{target_transcript.id}' contains no segments to index."
            job.error_details = "EmptyTranscriptError"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        active_chunker = chunker or SemanticTranscriptChunker()
        chunks = active_chunker.chunk_segments(
            segments=segments,
            meeting_id=meeting.id,
            organization_id=meeting.organization_id,
            transcript_id=target_transcript.id,
        )

        if not chunks:
            logger.warning(f"No chunks produced for transcript '{target_transcript.id}'.")
            job.status = JobStatus.COMPLETED.value
            job.progress = 100.0
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 7. Embedding Generation
        job.progress = 40.0
        db.commit()

        try:
            chunk_texts = [c.text for c in chunks]
            embeddings = configured_service.embed_passages(chunk_texts)
        except (ModelNotCachedError, EmbeddingModelLoadError) as mnc:
            logger.warning(
                f"Embedding model unavailable for job '{job_id}' on meeting '{meeting.id}': {mnc}"
            )
            job.status = JobStatus.FAILED.value
            job.error_message = "Embedding model is not cached locally and auto-download is disabled."
            job.error_details = "MODEL_NOT_CACHED"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return
        except Exception as emb_err:
            logger.exception(f"Embedding inference failed for job '{job_id}': {emb_err}")
            job.status = JobStatus.FAILED.value
            job.error_message = f"Embedding inference failed: {type(emb_err).__name__}"
            job.error_details = type(emb_err).__name__
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 8. Package Results
        items: List[ChunkEmbeddingResult] = []
        for c, emb in zip(chunks, embeddings):
            items.append(
                ChunkEmbeddingResult(
                    chunk=c,
                    embedding=emb,
                    dimension=dimension,
                    model_name=model_name,
                )
            )

        # 9. Transactional Storage & Stale Chunk Replacement
        job.progress = 70.0
        db.commit()

        active_storage = storage_manager or ChunkStorageManager(expected_dim=dimension)
        try:
            # Clean up old chunks for this transcript if re-indexing or replacing
            active_storage.delete_chunks_by_transcript(
                db=db,
                transcript_id=target_transcript.id,
                organization_id=meeting.organization_id,
            )
            # Store new chunks and synchronize FTS5
            active_storage.store_chunks(
                db=db,
                chunks_with_embeddings=items,
                organization_id=meeting.organization_id,
                transcript=target_transcript,
            )
            db.commit()
        except Exception as store_err:
            db.rollback()
            logger.exception(f"Failed to persist chunks for job '{job_id}': {store_err}")
            job = db.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.FAILED.value
                job.error_message = f"Failed to persist chunks: {type(store_err).__name__}"
                job.error_details = type(store_err).__name__
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    reconcile_meeting_status(db, job.meeting_id)
                except Exception:
                    pass
            return

        # 10. Complete Job Successfully
        job.status = JobStatus.COMPLETED.value
        job.progress = 100.0
        job.error_message = None
        job.error_details = None
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(
            f"Indexing job '{job_id}' completed successfully for meeting '{meeting.id}' "
            f"using transcript '{target_transcript.id}' ({len(items)} chunks indexed)."
        )
        # Reconcile overall meeting status after indexing completes
        try:
            reconcile_meeting_status(db, meeting.id)
        except Exception as re_exc:
            logger.warning(f"reconcile_meeting_status failed after indexing success: {re_exc}")

    except Exception as e:
        logger.exception(f"Fatal worker error in indexing job '{job_id}': {e}")
        try:
            db.rollback()
            job = db.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.FAILED.value
                job.error_message = f"Worker execution error: {type(e).__name__}"
                job.error_details = type(e).__name__
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                try:
                    reconcile_meeting_status(db, job.meeting_id)
                except Exception:
                    pass
        except Exception:
            pass
    finally:
        db.close()
