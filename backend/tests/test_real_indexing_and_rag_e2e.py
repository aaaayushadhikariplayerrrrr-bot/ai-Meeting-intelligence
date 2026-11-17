"""End-to-End Real Indexing Lifecycle and Real RAG Verification.

Verifies:
1. Audio generation -> Media Asset ingestion
2. Transcript creation -> Privacy cleaning and redaction
3. Automatic indexing job trigger -> Real E5 embedding generation on CPU
4. SQLite chunk/vector storage and FTS5 full-text synchronization
5. Strict privacy enforcement: unredacted content/PII is never indexed
6. Raw media deletion preserves indexed vectors and transcript segments
7. Authenticated RAG Query and Search APIs with real E5 vectors
8. Database citation verification with exact segment timestamps
9. Cross-organization isolation and honest provider attribution
"""

import os
import time
import wave
import struct
import json
import uuid
import numpy as np
import pytest
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.main import app
from app.database.session import SessionLocal
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus, MeetingStatus
from app.models.memory import TranscriptChunkRecord
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.core.security import create_access_token
from app.services.lifecycle import delete_raw_media_asset
from app.services.media_storage import get_meeting_storage_dir, to_safe_relative_path
from app.services.memory.embedding import MultilingualE5EmbeddingService, get_embedding_service
from app.services.memory.job_handler import process_indexing_job_sync
from app.services.memory.retrieval import search_hybrid_rrf, search_keyword
from app.services.memory.rag.engine import RAGEngine
from app.services.memory.rag.types import RAGQueryRequest, AbstentionStatus
from app.services.transcript_processing.processor import derive_processed_transcript


def _create_synthetic_wav(path: str, duration: float = 1.0) -> None:
    """Generate a valid 16kHz mono 16-bit PCM WAV file."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


@pytest.fixture
def real_e2e_setup(tmp_path):
    """Set up test organizations, users, and audio files."""
    db: Session = SessionLocal()
    try:
        # 1. Organizations
        org_alpha = Organization(id=str(uuid.uuid4()), name="E2E Real Org Alpha", slug=f"e2e-alpha-{uuid.uuid4().hex[:6]}")
        org_beta = Organization(id=str(uuid.uuid4()), name="E2E Real Org Beta", slug=f"e2e-beta-{uuid.uuid4().hex[:6]}")
        db.add_all([org_alpha, org_beta])
        db.commit()

        # 2. Users
        user_alpha = User(
            id=str(uuid.uuid4()),
            email=f"alpha_user_{uuid.uuid4().hex[:6]}@example.com",
            full_name="Alpha Engineer",
            hashed_password="hashed_pw_test",
            is_active=True,
        )
        user_beta = User(
            id=str(uuid.uuid4()),
            email=f"beta_user_{uuid.uuid4().hex[:6]}@example.com",
            full_name="Beta Director",
            hashed_password="hashed_pw_test",
            is_active=True,
        )
        db.add_all([user_alpha, user_beta])
        db.commit()

        # 3. Memberships
        mem_alpha = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=user_alpha.id,
            organization_id=org_alpha.id,
            role=OrgRole.ORG_ADMIN.value,
        )
        mem_beta = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=user_beta.id,
            organization_id=org_beta.id,
            role=OrgRole.ORG_ADMIN.value,
        )
        db.add_all([mem_alpha, mem_beta])
        db.commit()

        # 4. Auth Tokens
        token_alpha = create_access_token(subject=user_alpha.id, token_version=user_alpha.token_version)
        token_beta = create_access_token(subject=user_beta.id, token_version=user_beta.token_version)

        # 5. Audio File
        audio_file = str(tmp_path / "test_recording.wav")
        _create_synthetic_wav(audio_file, duration=2.0)

        yield {
            "db": db,
            "org_alpha": org_alpha,
            "org_beta": org_beta,
            "user_alpha": user_alpha,
            "user_beta": user_beta,
            "token_alpha": token_alpha,
            "token_beta": token_beta,
            "audio_file": audio_file,
            "tmp_path": tmp_path,
        }
    finally:
        db.close()


def test_real_indexing_lifecycle_and_rag_workflow(real_e2e_setup):
    """Complete end-to-end verification with real E5 model inference."""
    db: Session = real_e2e_setup["db"]
    org_alpha = real_e2e_setup["org_alpha"]
    org_beta = real_e2e_setup["org_beta"]
    user_alpha = real_e2e_setup["user_alpha"]
    audio_file = real_e2e_setup["audio_file"]

    # -------------------------------------------------------------------------
    # STAGE 1: Ingest Meeting and Media Asset
    # -------------------------------------------------------------------------
    meeting = Meeting(
        id=str(uuid.uuid4()),
        organization_id=org_alpha.id,
        created_by_user_id=user_alpha.id,
        title="Q3 Security & Database Infrastructure Review",
        status=MeetingStatus.COMPLETED.value,
    )
    db.add(meeting)
    db.commit()

    meeting_dir = get_meeting_storage_dir(meeting.id)
    audio_path = meeting_dir / "test_recording.wav"
    _create_synthetic_wav(str(audio_path), duration=2.0)
    storage_path = to_safe_relative_path(audio_path)

    media_asset = MediaAsset(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        storage_path=storage_path,
        file_name="test_recording.wav",
        file_size_bytes=os.path.getsize(str(audio_path)),
        media_type="AUDIO",
        mime_type="audio/wav",
    )
    db.add(media_asset)
    db.commit()
    assert os.path.exists(str(audio_path)), "Audio file must exist before processing"

    # -------------------------------------------------------------------------
    # STAGE 2: Persist ORIGINAL Transcript with Indian PII
    # -------------------------------------------------------------------------
    tr_original = Transcript(
        id=str(uuid.uuid4()),
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="en",
    )
    db.add(tr_original)
    db.commit()

    # Raw segments containing phone number and Aadhaar PII
    seg1 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=tr_original.id,
        sequence_order=1,
        start_time=0.0,
        end_time=12.5,
        text="The infrastructure team finalized deploying PostgreSQL version 16 on dedicated AWS RDS instances.",
        language_code="en",
    )
    seg2 = TranscriptSegment(
        id=str(uuid.uuid4()),
        transcript_id=tr_original.id,
        sequence_order=2,
        start_time=13.0,
        end_time=25.0,
        text="Lead architect contact number is 9876543210 and Aadhaar identity number is 2345 6789 0123 for background verification.",
        language_code="en",
    )
    db.add_all([seg1, seg2])
    db.commit()

    # -------------------------------------------------------------------------
    # STAGE 3: Privacy Cleaning & Redaction
    # -------------------------------------------------------------------------
    tr_safe = derive_processed_transcript(
        meeting_id=meeting.id,
        organization_id=org_alpha.id,
        target_type=TranscriptType.CLEANED_REDACTED,
        db=db,
        source_transcript_id=tr_original.id,
        user_id=user_alpha.id,
    )
    db.commit()

    # Verify PII is redacted in the derived transcript
    safe_segs = db.query(TranscriptSegment).filter_by(transcript_id=tr_safe.id).all()
    assert len(safe_segs) == 2
    full_safe_text = " ".join(s.text for s in safe_segs)
    assert "9876543210" not in full_safe_text, "Phone number must be redacted"
    assert "2345 6789 0123" not in full_safe_text, "Aadhaar must be redacted"
    assert "[PHONE_REDACTED]" in full_safe_text or "[AADHAAR_REDACTED]" in full_safe_text

    # -------------------------------------------------------------------------
    # STAGE 4: Verify Indexing Job Scheduled and Process with Real E5
    # -------------------------------------------------------------------------
    # Check that derive_processed_transcript auto-scheduled an EMBEDDING job
    indexing_job = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting.id,
            ProcessingJob.job_type == JobType.EMBEDDING.value,
        )
        .first()
    )
    assert indexing_job is not None, "Memory indexing job must be automatically scheduled"
    assert indexing_job.status in (JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value)

    # Execute the indexing job using the real cached E5 model
    service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
    assert service.is_cached()
    
    t0 = time.time()
    process_indexing_job_sync(job_id=indexing_job.id, embedding_service=service)
    indexing_duration = time.time() - t0
    
    db.refresh(indexing_job)
    assert indexing_job.status == JobStatus.COMPLETED.value, f"Job failed: {indexing_job.error_message}"
    assert indexing_job.progress == 100.0
    print(f"\n[E2E STAGE 4 PASS] Real E5 indexing completed in {indexing_duration:.2f}s.")

    # -------------------------------------------------------------------------
    # STAGE 5: Verify SQLite Chunks and FTS5 Storage
    # -------------------------------------------------------------------------
    chunks = (
        db.query(TranscriptChunkRecord)
        .filter_by(meeting_id=meeting.id, organization_id=org_alpha.id)
        .all()
    )
    assert len(chunks) > 0, "Chunks must be persisted in transcript_chunks"
    
    for c in chunks:
        # Verify 384-dimensional float32 vector
        assert c.dimension == 384
        assert len(c.embedding) == 384 * 4  # 1536 bytes
        vec = np.frombuffer(c.embedding, dtype=np.float32)
        assert np.all(np.isfinite(vec))
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-3, f"Vector must be unit normalized, norm={norm}"
        
        # Verify strict privacy: unredacted PII is NOT in chunk text
        assert "9876543210" not in c.text
        assert "2345 6789 0123" not in c.text
        
        # Verify chunks belong exclusively to the privacy-safe transcript
        assert c.transcript_id == tr_safe.id
        assert c.transcript_id != tr_original.id

    # Verify FTS5 Synchronization
    fts_matches = search_keyword(db, query="PostgreSQL", organization_id=org_alpha.id)
    assert len(fts_matches) > 0, "FTS5 must match 'PostgreSQL'"
    print(f"\n[E2E STAGE 5 PASS] Chunks and FTS5 synchronized. {len(chunks)} chunks created.")

    # -------------------------------------------------------------------------
    # STAGE 6: Verify Raw Audio Deletion Lifecycle
    # -------------------------------------------------------------------------
    # Raw media deletion must delete the disk file but preserve transcripts & chunks
    delete_res = delete_raw_media_asset(db=db, media_asset_id=media_asset.id, user_id=user_alpha.id)
    db.commit()
    assert delete_res.is_raw_deleted is True
    assert not os.path.exists(str(audio_path)), "Raw audio file on disk must be deleted"

    # Confirm chunk records and safe transcripts remain perfectly intact
    post_delete_chunks = (
        db.query(TranscriptChunkRecord)
        .filter_by(meeting_id=meeting.id, organization_id=org_alpha.id)
        .count()
    )
    assert post_delete_chunks == len(chunks), "Chunk records must survive raw audio deletion"
    print("\n[E2E STAGE 6 PASS] Raw audio safely deleted; chunks preserved.")

    # -------------------------------------------------------------------------
    # STAGE 7: Test Authenticated Search API with Real Vectors
    # -------------------------------------------------------------------------
    client = TestClient(app)
    headers_alpha = {
        "Authorization": f"Bearer {real_e2e_setup['token_alpha']}",
        "X-Organization-Id": org_alpha.id,
    }

    t_search_start = time.time()
    search_resp = client.post(
        "/api/v1/memory/search",
        headers=headers_alpha,
        json={"query": "PostgreSQL database deployment", "limit": 5},
    )
    search_latency = time.time() - t_search_start
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert search_data["total_results"] > 0
    top_search = search_data["results"][0]
    assert top_search["meeting_id"] == meeting.id
    assert "PostgreSQL" in top_search["text"]
    assert top_search["start_time"] == 0.0
    print(f"\n[E2E STAGE 7 PASS] Authenticated search succeeded in {search_latency*1000:.2f}ms.")

    # -------------------------------------------------------------------------
    # STAGE 8: Test Authenticated RAG Query API with Verified Database Citations
    # -------------------------------------------------------------------------
    t_query_start = time.time()
    query_resp = client.post(
        "/api/v1/memory/query",
        headers=headers_alpha,
        json={
            "question": "What database was finalized for deployment?",
            "max_evidence_chunks": 5,
        },
    )
    query_latency = time.time() - t_query_start
    assert query_resp.status_code == 200
    query_data = query_resp.json()

    assert query_data["is_abstained"] is False
    assert query_data["abstention_status"] == AbstentionStatus.ANSWER_SUPPORTED.value
    assert "PostgreSQL" in query_data["answer"]
    
    # Honest provider attribution check
    assert query_data["diagnostics"]["provider_used"] == "deterministic"
    assert query_data["diagnostics"]["is_heuristic"] is True

    # Citations check
    assert len(query_data["citations"]) > 0
    citation = query_data["citations"][0]
    assert citation["meeting_id"] == meeting.id
    assert citation["meeting_title"] == meeting.title
    assert citation["start_time"] == 0.0
    assert citation["end_time"] == 25.0
    assert "PostgreSQL" in citation["quoted_text"]
    print(f"\n[E2E STAGE 8 PASS] Authenticated RAG query succeeded in {query_latency*1000:.2f}ms.")
    print(f"  - Answer: {query_data['answer']}")
    print(f"  - Citation: [{citation['citation_id']}] {citation['meeting_title']} ({citation['start_time']}s - {citation['end_time']}s)")

    # -------------------------------------------------------------------------
    # STAGE 9: Test Unsupported Query (Abstention)
    # -------------------------------------------------------------------------
    unsupported_resp = client.post(
        "/api/v1/memory/query",
        headers=headers_alpha,
        json={
            "question": "What is the capital of Mars?",
            "max_evidence_chunks": 5,
        },
    )
    assert unsupported_resp.status_code == 200
    unsupported_data = unsupported_resp.json()
    assert unsupported_data["is_abstained"] is True
    assert unsupported_data["abstention_status"] in (
        AbstentionStatus.NO_EVIDENCE.value,
        AbstentionStatus.INSUFFICIENT_EVIDENCE.value,
    )
    assert len(unsupported_data["citations"]) == 0
    print(f"\n[E2E STAGE 9 PASS] Unsupported query abstained: {unsupported_data['abstention_status']}.")

    # -------------------------------------------------------------------------
    # STAGE 10: Strict Cross-Tenant Isolation
    # -------------------------------------------------------------------------
    headers_beta = {
        "Authorization": f"Bearer {real_e2e_setup['token_beta']}",
        "X-Organization-Id": org_beta.id,
    }
    beta_search_resp = client.post(
        "/api/v1/memory/search",
        headers=headers_beta,
        json={"query": "PostgreSQL database deployment", "limit": 5},
    )
    assert beta_search_resp.status_code == 200
    beta_data = beta_search_resp.json()
    assert beta_data["total_results"] == 0, "Org Beta must NOT see Org Alpha's PostgreSQL chunks"
    print("\n[E2E STAGE 10 PASS] Cross-tenant isolation verified: Org Beta returned 0 results.")
