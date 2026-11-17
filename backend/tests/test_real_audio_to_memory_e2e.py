"""End-to-End Real Audio -> Faster-Whisper GPU -> Privacy Redaction -> Real E5 CPU -> Memory Search Pipeline.

Verifies the complete unmocked pipeline:
1. Non-sensitive authorized spoken audio fixture (generated via Windows SAPI / System.Speech).
2. Live Faster-Whisper ASR inference on NVIDIA RTX 5060 Laptop GPU (CUDA float16).
3. Genuine transcription output persisted as ORIGINAL Transcript (no manual text insertion).
4. Privacy cleaning and redaction to CLEANED_REDACTED derived transcript.
5. Automatic scheduling and execution of EMBEDDING job with real E5 model on CPU.
6. SQLite BLOB vectors (384-dim, unit-normalized) and FTS5 full-text synchronization.
7. Verification that ORIGINAL transcript is strictly NOT indexed in memory.
8. Raw audio file deletion with complete preservation of chunks and transcripts.
9. Authenticated /api/v1/memory/search and /api/v1/memory/query retrieval.
"""

import os
import subprocess
import time
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
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.transcription.base import TranscriptionOptions
from app.services.transcript_processing.processor import derive_processed_transcript
from app.services.memory.embedding import MultilingualE5EmbeddingService
from app.services.memory.job_handler import process_indexing_job_sync
from app.services.memory.retrieval import search_keyword
from app.services.memory.rag.types import AbstentionStatus


def _synthesize_speech_audio(output_wav_path: str, spoken_text: str) -> None:
    """Generate a clean 16kHz WAV file of spoken English speech using Windows System.Speech."""
    ps_cmd = (
        f'Add-Type -AssemblyName System.Speech; '
        f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
        f'$synth.SetOutputToWaveFile("{output_wav_path}"); '
        f'$synth.Speak("{spoken_text}"); '
        f'$synth.Dispose()'
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=True)
    assert os.path.exists(output_wav_path), f"Audio fixture was not created at {output_wav_path}"
    assert os.path.getsize(output_wav_path) > 1000, "Audio fixture file is too small"


def test_real_audio_to_memory_e2e_pipeline():
    """Verify live audio -> GPU Whisper -> redaction -> CPU E5 -> SQLite/FTS5 -> search."""
    db: Session = SessionLocal()
    audio_path = None
    try:
        # 1. Setup Organization & User
        org = Organization(
            id=str(uuid.uuid4()),
            name="Audio E2E Engineering Org",
            slug=f"audio-org-{uuid.uuid4().hex[:6]}"
        )
        user = User(
            id=str(uuid.uuid4()),
            email=f"audio_dev_{uuid.uuid4().hex[:6]}@example.com",
            full_name="Audio Pipeline Engineer",
            is_active=True,
        )
        db.add_all([org, user])
        db.commit()

        membership = OrganizationMembership(
            id=str(uuid.uuid4()),
            user_id=user.id,
            organization_id=org.id,
            role=OrgRole.ORG_ADMIN.value,
        )
        db.add(membership)
        db.commit()

        token = create_access_token(subject=user.id, token_version=user.token_version)

        meeting = Meeting(
            id=str(uuid.uuid4()),
            organization_id=org.id,
            created_by_user_id=user.id,
            title="Live Audio Architecture Sync",
            status=MeetingStatus.COMPLETED.value,
        )
        db.add(meeting)
        db.commit()

        # 2. Generate Real Spoken Audio Fixture
        meeting_dir = get_meeting_storage_dir(meeting.id)
        audio_path = meeting_dir / "architecture_speech.wav"
        spoken_phrase = "The infrastructure team deployed PostgreSQL database on AWS."
        _synthesize_speech_audio(str(audio_path), spoken_phrase)

        storage_path = to_safe_relative_path(audio_path)
        media_asset = MediaAsset(
            id=str(uuid.uuid4()),
            meeting_id=meeting.id,
            storage_path=storage_path,
            file_name="architecture_speech.wav",
            file_size_bytes=os.path.getsize(str(audio_path)),
            media_type="AUDIO",
            mime_type="audio/wav",
        )
        db.add(media_asset)
        db.commit()

        # 3. Live Faster-Whisper Inference on NVIDIA RTX 5060 GPU
        # Uses small model cached locally
        asr_engine = FasterWhisperEngine(model_name="small", device="cuda", compute_type="float16")
        asr_engine.load_model()
        assert asr_engine.actual_device == "cuda", "Faster-Whisper must run on CUDA"

        asr_res = asr_engine.transcribe(
            str(audio_path),
            options=TranscriptionOptions(beam_size=1),
        )
        assert asr_res.device_used == "cuda"
        assert len(asr_res.segments) > 0, "ASR must produce at least one transcribed segment"

        transcribed_text = " ".join(s.text.strip() for s in asr_res.segments)
        print(f"\n[GPU ASR COMPLETE] Transcribed: \"{transcribed_text}\" on device={asr_res.device_used}")
        assert "postgresql" in transcribed_text.lower() or "database" in transcribed_text.lower(), (
            f"Expected PostgreSQL/database in transcription, got: {transcribed_text}"
        )

        # 4. Persist Genuine ORIGINAL Transcript
        tr_original = Transcript(
            id=str(uuid.uuid4()),
            meeting_id=meeting.id,
            transcript_type=TranscriptType.ORIGINAL.value,
            version=1,
            language_detected=asr_res.detected_language or "en",
            is_primary=True,
        )
        db.add(tr_original)
        db.commit()

        for seg in asr_res.segments:
            db_seg = TranscriptSegment(
                id=str(uuid.uuid4()),
                transcript_id=tr_original.id,
                sequence_order=seg.sequence_order,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text.strip(),
                language_code=seg.language_code or "en",
            )
            db.add(db_seg)
        db.commit()

        # 5. Derive CLEANED_REDACTED Transcript
        tr_safe = derive_processed_transcript(
            meeting_id=meeting.id,
            organization_id=org.id,
            target_type=TranscriptType.CLEANED_REDACTED,
            db=db,
            source_transcript_id=tr_original.id,
            user_id=user.id,
        )
        db.commit()
        assert tr_safe.transcript_type == TranscriptType.CLEANED_REDACTED.value

        # 6. Automatic EMBEDDING Job Triggered & Processed with Real E5 on CPU
        indexing_job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.meeting_id == meeting.id,
                ProcessingJob.job_type == JobType.EMBEDDING.value,
            )
            .first()
        )
        assert indexing_job is not None, "EMBEDDING job must be scheduled"

        e5_service = MultilingualE5EmbeddingService(device="cpu", allow_download=False)
        assert e5_service.is_cached()

        t0 = time.time()
        process_indexing_job_sync(job_id=indexing_job.id, embedding_service=e5_service)
        indexing_time = time.time() - t0

        db.refresh(indexing_job)
        assert indexing_job.status == JobStatus.COMPLETED.value
        print(f"[CPU E5 INDEXING COMPLETE] Indexed in {indexing_time:.2f}s on CPU.")

        # 7. Verify SQLite Vector BLOBs & FTS5 Synchronization
        chunks = (
            db.query(TranscriptChunkRecord)
            .filter_by(meeting_id=meeting.id, organization_id=org.id)
            .all()
        )
        assert len(chunks) > 0, "Chunks must exist in transcript_chunks"

        for c in chunks:
            # Must strictly belong to the safe transcript, never the original
            assert c.transcript_id == tr_safe.id
            assert c.transcript_id != tr_original.id
            assert c.dimension == 384
            vec = np.frombuffer(c.embedding, dtype=np.float32)
            assert np.all(np.isfinite(vec))
            norm = float(np.linalg.norm(vec))
            assert abs(norm - 1.0) < 1e-3

        # Verify FTS5 index contains transcribed keywords
        fts_hits = search_keyword(db, query="PostgreSQL", organization_id=org.id)
        assert len(fts_hits) > 0, "FTS5 must match 'PostgreSQL'"

        # 8. Raw Media Deletion Lifecycle Check
        del_asset = delete_raw_media_asset(db=db, media_asset_id=media_asset.id, user_id=user.id)
        db.commit()
        assert del_asset.is_raw_deleted is True
        assert not os.path.exists(str(audio_path)), "Raw audio file must be deleted from disk"

        # Ensure chunks and safe segments survive raw audio deletion
        post_del_chunks = db.query(TranscriptChunkRecord).filter_by(meeting_id=meeting.id).count()
        assert post_del_chunks == len(chunks), "Indexed chunks must survive audio deletion"

        # 9. Authenticated Search API with Real E5 Vector
        client = TestClient(app)
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Organization-Id": org.id,
        }
        search_resp = client.post(
            "/api/v1/memory/search",
            headers=headers,
            json={"query": "Where did the team deploy PostgreSQL?", "limit": 5},
        )
        assert search_resp.status_code == 200
        search_json = search_resp.json()
        assert search_json["total_results"] > 0
        top_res = search_json["results"][0]
        assert top_res["meeting_id"] == meeting.id
        assert "postgresql" in top_res["text"].lower()

        # 10. Authenticated Ask AI Query API
        query_resp = client.post(
            "/api/v1/memory/query",
            headers=headers,
            json={"question": "What database was deployed?", "max_evidence_chunks": 5},
        )
        assert query_resp.status_code == 200
        query_json = query_resp.json()
        assert query_json["is_abstained"] is False
        assert query_json["abstention_status"] == AbstentionStatus.ANSWER_SUPPORTED.value
        assert "PostgreSQL" in query_json["answer"] or "postgresql" in query_json["answer"].lower()
        assert query_json["diagnostics"]["provider_used"] == "deterministic"
        assert len(query_json["citations"]) > 0

        citation = query_json["citations"][0]
        assert citation["meeting_id"] == meeting.id
        assert citation["start_time"] >= 0.0

        print("[FULL REAL PIPELINE PASS] Audio -> Whisper GPU -> Redaction -> E5 CPU -> Memory Search!")

    finally:
        db.close()
        if audio_path and os.path.exists(str(audio_path)):
            try:
                os.remove(str(audio_path))
            except Exception:
                pass
