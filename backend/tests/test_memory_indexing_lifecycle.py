"""Comprehensive tests for Phase 8.3: Automatic Privacy-Safe Memory Indexing & Lifecycle Integration."""

import json
import uuid
from typing import List
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.identity import Organization, User
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.schemas.memory import IndexingJobPayload
from app.services.job_worker import (
    claim_job_atomically,
    execute_claimed_job,
    reconcile_stranded_jobs_on_startup,
)
from app.services.lifecycle import delete_raw_media_asset
from app.services.memory.exceptions import (
    ModelNotCachedError,
    PrivacyEnforcementError,
    TenantSecurityError,
)
from app.services.memory.job_handler import (
    schedule_indexing_job,
    process_indexing_job_sync,
)
from app.services.memory.retrieval import search_hybrid_rrf, search_keyword
from app.services.memory.storage import ChunkStorageManager, EXPECTED_DIMENSION
from app.services.transcript_processing.processor import derive_processed_transcript


def make_synthetic_vector(seed: int, dim: int = EXPECTED_DIMENSION) -> List[float]:
    """Generate a deterministic unit-normalized 384-dimensional synthetic vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist()


class MockEmbeddingService:
    """Mock embedding service producing deterministic vectors without external model downloads."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return [make_synthetic_vector(seed=100 + i, dim=self.dimension) for i, _ in enumerate(texts)]


@pytest.fixture(autouse=True)
def bind_session_local_to_test_engine(db_session, monkeypatch):
    """Ensure worker and memory handlers use the test database session without closing it."""
    monkeypatch.setattr("app.database.session.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.job_worker.SessionLocal", lambda: db_session)
    monkeypatch.setattr("app.services.memory.job_handler.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)


@pytest.fixture
def indexing_lifecycle_setup(db_session: Session):
    """Fixture providing isolated Organizations, Meetings, Users, and Transcripts."""
    org_a = Organization(name="Indexing Org A", slug=f"idx-org-a-{uuid.uuid4().hex[:8]}")
    org_b = Organization(name="Indexing Org B", slug=f"idx-org-b-{uuid.uuid4().hex[:8]}")
    user_a = User(
        email=f"user_a_{uuid.uuid4().hex[:8]}@example.com",
        full_name="User A",
        hashed_password="pw",
        is_active=True,
    )
    db_session.add_all([org_a, org_b, user_a])
    db_session.commit()

    meeting_a = Meeting(
        organization_id=org_a.id,
        created_by_user_id=user_a.id,
        title="Architecture Discussion",
        status=MeetingStatus.COMPLETED.value,
    )
    meeting_b = Meeting(
        organization_id=org_b.id,
        title="Org B Secret Meeting",
        status=MeetingStatus.COMPLETED.value,
    )
    db_session.add_all([meeting_a, meeting_b])
    db_session.commit()

    # Original unredacted transcript for meeting A
    t_orig = Transcript(
        meeting_id=meeting_a.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
    )
    db_session.add(t_orig)
    db_session.commit()

    segs = [
        TranscriptSegment(
            transcript_id=t_orig.id,
            sequence_order=0,
            start_time=0.0,
            end_time=5.0,
            text="Hello team, welcome to the quarterly architecture review.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=t_orig.id,
            sequence_order=1,
            start_time=5.0,
            end_time=12.0,
            text="Contact developer at secret-api-key-12345 or user@example.com.",
            language_code="en",
        ),
        TranscriptSegment(
            transcript_id=t_orig.id,
            sequence_order=2,
            start_time=12.0,
            end_time=20.0,
            text="અમે નવા ડેટાબેઝ માઇગ્રેશન પર કામ કરી રહ્યા છીએ.",  # Gujarati
            language_code="gu",
        ),
    ]
    db_session.add_all(segs)
    db_session.commit()

    return {
        "org_a": org_a,
        "org_b": org_b,
        "user_a": user_a,
        "meeting_a": meeting_a,
        "meeting_b": meeting_b,
        "t_orig": t_orig,
    }


# ============================================================================
# 1. Automatic Indexing Trigger & Derivation Tests
# ============================================================================

def test_automatic_indexing_scheduled_after_redaction_persistence(db_session: Session, indexing_lifecycle_setup):
    """Verify that committing a CLEANED_REDACTED transcript automatically schedules a background indexing job."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    # Derive CLEANED_REDACTED transcript
    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    # Check that an indexing job was automatically scheduled
    job = (
        db_session.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_a.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .first()
    )
    assert job is not None
    assert job.status == JobStatus.PENDING.value
    assert job.progress == 0.0

    payload = json.loads(job.payload)
    assert payload["meeting_id"] == meeting_a.id
    assert payload["organization_id"] == org_a.id
    assert payload["transcript_id"] == derived.id


def test_no_indexing_before_redaction_completes(db_session: Session, indexing_lifecycle_setup):
    """Verify that generating an unredacted CLEANED transcript does NOT schedule an indexing job."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    # Derive unredacted CLEANED transcript
    derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED,
        db=db_session,
    )

    # No EMBEDDING job should be scheduled
    job_count = (
        db_session.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_a.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .count()
    )
    assert job_count == 0


def test_original_and_cleaned_rejection_in_scheduling(db_session: Session, indexing_lifecycle_setup):
    """Strict Privacy: Never permit scheduling indexing for ORIGINAL or CLEANED transcripts."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]
    t_orig = setup["t_orig"]

    # Attempt scheduling with ORIGINAL transcript
    with pytest.raises(PrivacyEnforcementError, match="unredacted transcript type"):
        schedule_indexing_job(
            db=db_session,
            meeting_id=meeting_a.id,
            organization_id=org_a.id,
            transcript_id=t_orig.id,
        )

    # Derive CLEANED transcript and attempt scheduling
    t_cleaned = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED,
        db=db_session,
    )
    with pytest.raises(PrivacyEnforcementError, match="unredacted transcript type"):
        schedule_indexing_job(
            db=db_session,
            meeting_id=meeting_a.id,
            organization_id=org_a.id,
            transcript_id=t_cleaned.id,
        )


def test_duplicate_indexing_job_prevention(db_session: Session, indexing_lifecycle_setup):
    """Verify calling schedule_indexing_job multiple times reuses the existing active job without duplicates."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )

    # First call scheduled it automatically in derive_processed_transcript
    # Second manual call:
    job2 = schedule_indexing_job(
        db=db_session,
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        transcript_id=derived.id,
    )

    all_jobs = (
        db_session.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_a.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .all()
    )
    assert len(all_jobs) == 1
    assert all_jobs[0].id == job2.id


def test_organization_ownership_validation_in_scheduling(db_session: Session, indexing_lifecycle_setup):
    """Tenant Security: Cannot schedule indexing across organization boundaries."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_b = setup["org_b"]

    with pytest.raises(TenantSecurityError):
        schedule_indexing_job(
            db=db_session,
            meeting_id=meeting_a.id,
            organization_id=org_b.id,
        )


# ============================================================================
# 2. Execution, Chunking, Embedding & Searchability Tests
# ============================================================================

def test_successful_indexing_execution_and_searchability(db_session: Session, indexing_lifecycle_setup):
    """Verify end-to-end execution of background indexing job and resulting hybrid searchability."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    job = (
        db_session.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_a.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .first()
    )
    assert job is not None

    # Execute with mock embedding service
    mock_service = MockEmbeddingService()
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job.id, embedding_service=mock_service)

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0
    assert job.error_message is None

    # Verify chunks exist in transcript_chunks table
    chunks = (
        db_session.query(TranscriptChunkRecord)
        .filter(TranscriptChunkRecord.transcript_id == derived.id)
        .all()
    )
    assert len(chunks) >= 1
    assert chunks[0].organization_id == org_a.id
    assert chunks[0].meeting_id == meeting_a.id

    # Verify searchable via keyword and hybrid retrieval
    kw_results = search_keyword(db_session, query="architecture review", organization_id=org_a.id)
    assert len(kw_results) >= 1

    hybrid_results = search_hybrid_rrf(
        db=db_session,
        query="architecture",
        query_vector=make_synthetic_vector(seed=100),
        organization_id=org_a.id,
    )
    assert len(hybrid_results) >= 1
    assert hybrid_results[0].organization_id == org_a.id
    assert hybrid_results[0].rrf_score > 0.0


def test_model_not_cached_failure_behavior(db_session: Session, indexing_lifecycle_setup):
    """Verify that when the embedding model is not cached, the job fails gracefully without fake vectors."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )

    job = (
        db_session.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting_a.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .first()
    )

    # Failing mock raising ModelNotCachedError
    failing_service = MagicMock()
    failing_service.model_name = "intfloat/multilingual-e5-small"
    failing_service.dimension = 384
    failing_service.embed_passages.side_effect = ModelNotCachedError("E5 model not in local cache")

    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job.id, embedding_service=failing_service)

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "MODEL_NOT_CACHED"
    assert "not cached locally" in job.error_message

    # Verify ZERO chunks or fake embeddings were created
    chunk_count = (
        db_session.query(TranscriptChunkRecord)
        .filter(TranscriptChunkRecord.transcript_id == derived.id)
        .count()
    )
    assert chunk_count == 0

    # Verify transcript remains 100% intact
    db_session.refresh(derived)
    assert len(derived.segments) == 3


def test_idempotent_indexing_avoids_redundant_inference(db_session: Session, indexing_lifecycle_setup):
    """Verify indexing an already-indexed transcript is idempotent and avoids redundant embedding."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )
    job1 = (
        db_session.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
        .first()
    )

    mock_service = MockEmbeddingService()
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job1.id, embedding_service=mock_service)

    initial_chunks = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).count()
    assert initial_chunks >= 1

    # Second indexing job for the same transcript without force_reindex
    payload2 = IndexingJobPayload(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        transcript_id=derived.id,
        force_reindex=False,
    )
    job2 = ProcessingJob(
        meeting_id=meeting_a.id,
        job_type=JobType.EMBEDDING.value,
        status=JobStatus.PENDING.value,
        payload=json.dumps(payload2.model_dump()),
    )
    db_session.add(job2)
    db_session.commit()

    spy_service = MagicMock()
    spy_service.model_name = "intfloat/multilingual-e5-small"
    spy_service.dimension = 384

    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job2.id, embedding_service=spy_service)

    db_session.refresh(job2)
    assert job2.status == JobStatus.COMPLETED.value
    # embed_passages should NOT have been called due to idempotency check
    spy_service.embed_passages.assert_not_called()

    # Chunk count did not change
    after_chunks = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).count()
    assert after_chunks == initial_chunks


def test_reprocessing_replaces_stale_chunks(db_session: Session, indexing_lifecycle_setup):
    """Verify force re-indexing cleanly replaces stale chunks without duplicates."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )
    job1 = (
        db_session.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
        .first()
    )

    mock_service = MockEmbeddingService()
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job1.id, embedding_service=mock_service)

    chunks_v1 = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).count()
    assert chunks_v1 >= 1

    # Force reindex
    job2 = schedule_indexing_job(
        db=db_session,
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        transcript_id=derived.id,
        force_reindex=True,
    )
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job2.id, embedding_service=mock_service)

    db_session.refresh(job2)
    assert job2.status == JobStatus.COMPLETED.value

    # Chunk count is unchanged (no duplicates created)
    chunks_after = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).count()
    assert chunks_after == chunks_v1


# ============================================================================
# 3. Worker Integration & Startup Recovery Tests
# ============================================================================

def test_worker_dispatch_executes_embedding_job(db_session: Session, indexing_lifecycle_setup):
    """Verify the single-worker event dispatch loop picks up and executes EMBEDDING jobs."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.REDACTED,
        db=db_session,
    )
    job = (
        db_session.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
        .first()
    )
    assert job.status == JobStatus.PENDING.value

    # Atomically claim job
    claimed = claim_job_atomically(db_session, job.id)
    assert claimed is not None
    assert claimed.status == JobStatus.IN_PROGRESS.value

    # Execute claimed job with mocked embedding service
    with patch("app.services.job_worker.SessionLocal", lambda: db_session), \
         patch("app.services.memory.job_handler.SessionLocal", lambda: db_session), \
         patch("app.services.memory.job_handler.get_embedding_service", return_value=MockEmbeddingService()):
        execute_claimed_job(claimed.id)

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0


def test_interrupted_indexing_recovery_on_startup(db_session: Session, indexing_lifecycle_setup):
    """Verify startup recovery marks stranded EMBEDDING jobs as FAILED without breaking completed meeting status."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    assert meeting_a.status == MeetingStatus.COMPLETED.value

    # Create an IN_PROGRESS stranded job
    job = ProcessingJob(
        meeting_id=meeting_a.id,
        job_type=JobType.EMBEDDING.value,
        status=JobStatus.IN_PROGRESS.value,
        progress=50.0,
        payload=json.dumps({"meeting_id": meeting_a.id, "organization_id": setup["org_a"].id}),
    )
    db_session.add(job)
    db_session.commit()

    # Reconcile on startup
    reconciled = reconcile_stranded_jobs_on_startup(db_session)
    assert reconciled >= 1

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert job.error_details == "ServerRestartInterruption"

    # Meeting status remains COMPLETED (transcription is intact)
    db_session.refresh(meeting_a)
    assert meeting_a.status == MeetingStatus.COMPLETED.value


# ============================================================================
# 4. Deletion Synchronization & Media Retention Tests
# ============================================================================

def test_transcript_deletion_synchronization(db_session: Session, indexing_lifecycle_setup):
    """Verify deleting a transcript automatically cascades to delete associated chunks and FTS5 entries."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )
    job = (
        db_session.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
        .first()
    )
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job.id, embedding_service=MockEmbeddingService())

    # Chunks exist
    chunks = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).all()
    assert len(chunks) >= 1
    cid = chunks[0].id

    # Verify FTS entry exists
    fts_before = db_session.execute(text("SELECT count(*) FROM transcript_chunks_fts WHERE chunk_id = :cid"), {"cid": cid}).scalar()
    assert fts_before == 1

    # Delete transcript
    db_session.delete(derived)
    db_session.commit()

    # Chunks are gone
    chunks_after = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.transcript_id == derived.id).count()
    assert chunks_after == 0

    # FTS entry is gone
    fts_after = db_session.execute(text("SELECT count(*) FROM transcript_chunks_fts WHERE chunk_id = :cid"), {"cid": cid}).scalar()
    assert fts_after == 0


def test_meeting_deletion_synchronization(db_session: Session, indexing_lifecycle_setup):
    """Verify deleting a meeting completely cascades and cleans up all chunks and FTS5 entries."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    derived = derive_processed_transcript(
        meeting_id=meeting_a.id,
        organization_id=org_a.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db_session,
    )
    job = (
        db_session.query(ProcessingJob)
        .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
        .first()
    )
    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job.id, embedding_service=MockEmbeddingService())

    chunk_ids = [c.id for c in db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == meeting_a.id).all()]
    assert len(chunk_ids) >= 1

    # Delete meeting
    db_session.delete(meeting_a)
    db_session.commit()

    # Verify all chunks and FTS entries are purged
    remaining_chunks = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == meeting_a.id).count()
    assert remaining_chunks == 0

    for cid in chunk_ids:
        fts_cnt = db_session.execute(text("SELECT count(*) FROM transcript_chunks_fts WHERE chunk_id = :cid"), {"cid": cid}).scalar()
        assert fts_cnt == 0


def test_raw_audio_deletion_does_not_delete_memory(db_session: Session, indexing_lifecycle_setup):
    """Verify raw audio deletion does NOT delete organizational memory or privacy-safe chunks."""
    from pathlib import Path
    from app.core.config import settings

    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]
    user_a = setup["user_a"]

    # Create dummy raw media asset relative to settings.STORAGE_DIR
    rel_path = f"test_raw_{meeting_a.id}.wav"
    full_path = Path(settings.STORAGE_DIR) / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(b"RIFF dummy wav data")

    try:
        media_asset = MediaAsset(
            meeting_id=meeting_a.id,
            file_name="test_raw_audio.wav",
            storage_path=rel_path,
            media_type="AUDIO",
            mime_type="audio/wav",
            is_raw_deleted=False,
        )
        db_session.add(media_asset)
        db_session.commit()

        # Index transcript
        derived = derive_processed_transcript(
            meeting_id=meeting_a.id,
            organization_id=org_a.id,
            target_type=TranscriptType.CLEANED_REDACTED,
            db=db_session,
        )
        job = (
            db_session.query(ProcessingJob)
            .filter(ProcessingJob.meeting_id == meeting_a.id, ProcessingJob.job_type == JobType.EMBEDDING.value)
            .first()
        )
        process_indexing_job_sync(job_id=job.id, embedding_service=MockEmbeddingService())

        chunks_before = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == meeting_a.id).count()
        assert chunks_before >= 1

        # Delete raw media asset
        delete_raw_media_asset(
            db=db_session,
            media_asset_id=media_asset.id,
            user_id=user_a.id,
            delete_normalized=False,
        )

        db_session.refresh(media_asset)
        assert media_asset.is_raw_deleted is True

        # Memory and chunks remain 100% intact and searchable
        chunks_after = db_session.query(TranscriptChunkRecord).filter(TranscriptChunkRecord.meeting_id == meeting_a.id).count()
        assert chunks_after == chunks_before

        search_res = search_keyword(db_session, query="architecture", organization_id=org_a.id)
        assert len(search_res) >= 1
    finally:
        if full_path.exists():
            full_path.unlink()


def test_no_pii_or_transcript_text_in_job_errors(db_session: Session, indexing_lifecycle_setup):
    """Privacy Verification: Failed indexing job records only sanitized error codes, zero transcript text or PII."""
    setup = indexing_lifecycle_setup
    meeting_a = setup["meeting_a"]
    org_a = setup["org_a"]

    # Create job with invalid payload
    job = ProcessingJob(
        meeting_id=meeting_a.id,
        job_type=JobType.EMBEDDING.value,
        status=JobStatus.PENDING.value,
        payload="invalid non-json string with sensitive token secret_12345",
    )
    db_session.add(job)
    db_session.commit()

    with patch("app.services.memory.job_handler.SessionLocal", lambda: db_session):
        process_indexing_job_sync(job_id=job.id)

    db_session.refresh(job)
    assert job.status == JobStatus.FAILED.value
    assert "secret_12345" not in (job.error_message or "")
    assert "secret_12345" not in (job.error_details or "")
