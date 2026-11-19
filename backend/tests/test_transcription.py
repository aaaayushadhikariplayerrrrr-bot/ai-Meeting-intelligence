"""Comprehensive tests for Multilingual ASR Pipeline and Speech Recognition Architecture.

Tests cover:
- GPU hardware detection and CTranslate2 inspection
- Dynamic device selection and automatic CPU fallback
- Audio chunk calculation, WAV slicing, and overlap deduplication
- Model-independent ASREngine interface and provider registry
- Decoupled Diarization provider and specialist adapter stubs
- Database persistence of Transcripts and TranscriptSegments
- Original transcript preservation and multi-version coexistence
- API endpoints for starting jobs, retrieving transcripts, and hardware status
"""

import os
import wave
import struct
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app as fastapi_app
from app.models.identity import Organization, User, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobType, JobStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.transcription.base import ASRSegment, ASRResult, TranscriptionOptions
from app.services.transcription.gpu_manager import (
    check_gpu_hardware,
    check_ctranslate2_support,
    select_optimal_device_and_compute,
)
from app.services.transcription.chunking import (
    compute_audio_chunks,
    extract_wav_chunk,
    merge_and_deduplicate_segments,
)
from app.services.transcription.diarization import NullDiarizationProvider
from app.services.transcription.specialist_adapters import (
    IndicConformerEngine,
    WhisperHindi2HinglishApexEngine,
)
from app.services.transcription.faster_whisper_engine import FasterWhisperEngine
from app.services.transcription.job_handler import process_transcription_job_sync
from app.core.config import settings


def _create_synthetic_16k_mono_wav(path: str, duration_sec: float = 1.0) -> None:
    """Create a valid 16 kHz 16-bit mono PCM WAV for testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        # Generate silence
        data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
        wav.writeframes(data)


# --- 1. GPU & Device Selection Tests ---

def test_gpu_hardware_inspection():
    """Verify GPU hardware diagnostics return required keys."""
    hw = check_gpu_hardware()
    assert "torch_cuda_available" in hw
    assert "device_count" in hw
    assert "total_vram_mb" in hw
    assert "free_vram_mb" in hw


def test_device_selection_cpu_explicit():
    """Verify explicit CPU preference returns cpu/int8."""
    dev, comp, reason = select_optimal_device_and_compute(preferred_device="cpu")
    assert dev == "cpu"
    assert comp == "int8"
    assert "explicitly requested" in reason


def test_device_selection_low_vram_fallback(monkeypatch):
    """Verify fallback to CPU when available VRAM is below threshold."""
    monkeypatch.setattr(
        "app.services.transcription.gpu_manager.check_ctranslate2_support",
        lambda: (True, "Ready", ["float16", "int8"])
    )
    monkeypatch.setattr(
        "app.services.transcription.gpu_manager.check_gpu_hardware",
        lambda: {"free_vram_mb": 500.0, "total_vram_mb": 8000.0, "device_name": "Test GPU"}
    )
    dev, comp, reason = select_optimal_device_and_compute(
        preferred_device="cuda",
        min_free_vram_mb=2000.0
    )
    assert dev == "cpu"
    assert comp == "int8"
    assert "Low VRAM" in reason


# --- 2. Chunking & Timestamp Mathematics Tests ---

def test_compute_audio_chunks_short_audio():
    """Short audio (<= chunk_length) should produce a single chunk."""
    chunks = compute_audio_chunks(total_duration=120.0, chunk_length=300.0, overlap=2.0)
    assert len(chunks) == 1
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 120.0
    assert chunks[0].is_last is True


def test_compute_audio_chunks_long_audio():
    """Long audio (> chunk_length) should produce overlapping chunks."""
    # 700s audio with 300s chunk and 2s overlap:
    # Chunk 0: 0 to 300
    # Chunk 1: 298 to 598
    # Chunk 2: 596 to 700 (last)
    chunks = compute_audio_chunks(total_duration=700.0, chunk_length=300.0, overlap=2.0)
    assert len(chunks) == 3
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 300.0
    assert chunks[1].start_time == 298.0
    assert chunks[1].end_time == 598.0
    assert chunks[2].start_time == 596.0
    assert chunks[2].end_time == 700.0
    assert chunks[2].is_last is True


def test_extract_wav_chunk(tmp_path):
    """Verify lossless WAV slicing from normalized audio."""
    src_wav = str(tmp_path / "source.wav")
    _create_synthetic_16k_mono_wav(src_wav, duration_sec=3.0)

    chunk_wav = str(tmp_path / "chunk.wav")
    extract_wav_chunk(src_wav, chunk_wav, start_time=1.0, end_time=2.5)

    with wave.open(chunk_wav, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        # 1.5 seconds * 16000 = 24000 frames
        assert w.getnframes() == 24000


def test_merge_and_deduplicate_segments():
    """Verify segment merging correctly computes absolute offsets and deduplicates boundary overlaps."""
    from app.services.transcription.chunking import AudioChunk

    chunk0 = AudioChunk(index=0, start_time=0.0, end_time=300.0, is_last=False)
    chunk1 = AudioChunk(index=1, start_time=298.0, end_time=600.0, is_last=True)

    segs0 = [
        ASRSegment(sequence_order=0, start_time=10.0, end_time=15.0, text="Namaste", language_code="hi"),
        ASRSegment(sequence_order=1, start_time=295.0, end_time=299.5, text="Meeting starts now", language_code="en"),
    ]
    segs1 = [
        # Duplicate of previous boundary utterance
        ASRSegment(sequence_order=0, start_time=0.0, end_time=1.5, text="Meeting starts now", language_code="en"),
        # New utterance in second chunk (local 5.0 to 10.0 -> abs 303.0 to 308.0)
        ASRSegment(sequence_order=1, start_time=5.0, end_time=10.0, text="Aavjo", language_code="gu"),
    ]

    merged = merge_and_deduplicate_segments([(chunk0, segs0), (chunk1, segs1)], overlap=2.0)
    assert len(merged) == 3
    assert merged[0].text == "Namaste"
    assert merged[0].start_time == 10.0
    assert merged[1].text == "Meeting starts now"
    assert merged[1].start_time == 295.0
    assert merged[2].text == "Aavjo"
    assert merged[2].start_time == 303.0
    assert merged[2].language_code == "gu"
    assert [m.sequence_order for m in merged] == [0, 1, 2]


# --- 3. Diarization & Specialist Adapters Tests ---

def test_null_diarization_provider():
    """Verify NullDiarizationProvider does not invent speakers."""
    provider = NullDiarizationProvider()
    assert provider.is_available() is False
    res = provider.diarize("dummy.wav")
    assert res.speaker_count == 0
    assert len(res.turns) == 0


def test_specialist_adapters_contract():
    """Verify future specialist ASR adapters conform to ASREngine contract."""
    conformer = IndicConformerEngine()
    info = conformer.get_engine_info()
    assert info["engine_name"] == "ai4bharat-indic-conformer"
    assert "hi" in info["supported_languages"]
    assert "gu" in info["supported_languages"]
    assert info["installed"] is False

    apex = WhisperHindi2HinglishApexEngine()
    info_apex = apex.get_engine_info()
    assert info_apex["engine_name"] == "oriserve-whisper-hindi2hinglish-apex"
    assert "hi-en" in info_apex["supported_languages"]


# --- 4. Database Persistence & Version Preservation Tests ---

def test_transcription_persistence_and_versioning(db_session, tmp_path, monkeypatch):
    """Verify transcription persistence in DB, preserving versions without overwriting."""
    # 1. Create test Org & Meeting
    org = Organization(name="ASR Test Org", slug="asr-test-org")
    db_session.add(org)
    db_session.commit()

    meeting = Meeting(organization_id=org.id, title="Multilingual Sprint Sync")
    db_session.add(meeting)
    db_session.commit()

    # 2. Create audio file on disk and MediaAsset
    audio_file = tmp_path / "normalized.wav"
    _create_synthetic_16k_mono_wav(str(audio_file), duration_sec=2.0)

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="normalized.wav",
        storage_path=str(audio_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        duration_seconds=2.0,
    )
    db_session.add(asset)
    db_session.commit()

    # 3. Create ProcessingJob
    job1 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job1)
    db_session.commit()

    # 4. Mock ASR engine transcription output
    mock_result1 = ASRResult(
        segments=[
            ASRSegment(sequence_order=0, start_time=0.0, end_time=1.0, text="Aapka swagat hai", language_code="hi"),
            ASRSegment(sequence_order=1, start_time=1.0, end_time=2.0, text="Let's begin the review", language_code="en"),
        ],
        full_text="Aapka swagat hai Let's begin the review",
        detected_language="hi",
        language_probability=0.98,
        duration_seconds=2.0,
        device_used="cuda:0",
        compute_type_used="float16",
        model_name="large-v3-turbo",
        engine_name="faster-whisper",
    )

    class MockEngine:
        def transcribe(self, *args, **kwargs):
            return mock_result1

    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: MockEngine())
    monkeypatch.setattr("app.services.transcription.job_handler.SessionLocal", lambda: db_session)
    # Avoid auto-closing db_session in test
    monkeypatch.setattr(db_session, "close", lambda: None)

    # 5. Execute first transcription job
    process_transcription_job_sync(job1.id)

    # Verify first transcript: version 1, is_primary True
    db_session.refresh(job1)
    assert job1.status == JobStatus.COMPLETED.value
    assert job1.progress == 100.0

    t1 = db_session.query(Transcript).filter(Transcript.meeting_id == meeting.id).first()
    assert t1 is not None
    assert t1.version == 1
    assert t1.is_primary is True
    assert t1.language_detected == "hi"
    assert len(t1.segments) == 2
    assert t1.segments[0].text == "Aapka swagat hai"
    assert t1.segments[0].language_code == "hi"
    assert t1.segments[1].text == "Let's begin the review"
    assert t1.segments[1].language_code == "en"

    # 6. Execute second transcription job (re-run)
    job2 = ProcessingJob(
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        status=JobStatus.PENDING.value,
    )
    db_session.add(job2)
    db_session.commit()

    mock_result2 = ASRResult(
        segments=[
            ASRSegment(sequence_order=0, start_time=0.0, end_time=2.0, text="Updated transcription text", language_code="en"),
        ],
        full_text="Updated transcription text",
        detected_language="en",
        language_probability=0.99,
        duration_seconds=2.0,
        device_used="cpu",
        compute_type_used="int8",
        model_name="large-v3-turbo",
        engine_name="faster-whisper",
    )
    class MockEngine2:
        def transcribe(self, *args, **kwargs):
            return mock_result2

    monkeypatch.setattr("app.services.transcription.job_handler.get_asr_engine", lambda: MockEngine2())

    process_transcription_job_sync(job2.id)

    # Verify BOTH transcripts coexist and original was NOT overwritten
    all_transcripts = (
        db_session.query(Transcript)
        .filter(Transcript.meeting_id == meeting.id)
        .order_by(Transcript.version.asc())
        .all()
    )
    assert len(all_transcripts) == 2
    assert all_transcripts[0].version == 1
    assert all_transcripts[0].is_primary is True
    assert len(all_transcripts[0].segments) == 2
    assert all_transcripts[1].version == 2
    assert all_transcripts[1].is_primary is False
    assert len(all_transcripts[1].segments) == 1


# --- 5. API Endpoint Tests ---

def test_api_hardware_diagnostics(client: TestClient, super_admin_no_membership, monkeypatch):
    """Verify /api/v1/transcription/hardware/diagnostics behavior:
    1. Gated with 403 when ENABLE_SYSTEM_DIAGNOSTICS is False (default).
    2. Returns full hardware report for Super Admin when ENABLE_SYSTEM_DIAGNOSTICS is True.
    """
    from app.core.config import settings
    from backend.tests.conftest import make_auth_headers
    headers = make_auth_headers(super_admin_no_membership)

    # 1. Disabled by default -> 403
    resp_disabled = client.get("/api/v1/transcription/hardware/diagnostics", headers=headers)
    assert resp_disabled.status_code == 403
    assert "System and hardware diagnostics are disabled" in resp_disabled.json()["detail"]

    # 2. Enabled explicitly -> 200
    monkeypatch.setattr(settings, "ENABLE_SYSTEM_DIAGNOSTICS", True)
    resp = client.get("/api/v1/transcription/hardware/diagnostics", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "gpu_info" in data
    assert "ctranslate2_cuda_support" in data
    assert "active_engine" in data
    assert data["active_engine"]["engine_name"] == "faster-whisper"


def test_api_start_transcription_and_retrieve(client: TestClient, db_session, tmp_path, monkeypatch):
    """Test full API lifecycle: start transcription, poll job, retrieve segments."""
    from backend.tests.conftest import make_auth_headers
    from app.core.security import hash_password

    org = Organization(name="Default Organization", slug="default-org", is_active=True)
    db_session.add(org)
    db_session.flush()

    user = User(
        email="transcribe_admin@example.com",
        full_name="Transcribe Admin",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    membership = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(membership)

    meeting = Meeting(organization_id=org.id, title="API Test Meeting")
    db_session.add(meeting)
    db_session.commit()

    headers = make_auth_headers(user, org)

    audio_file = tmp_path / "api_test.wav"
    _create_synthetic_16k_mono_wav(str(audio_file), duration_sec=1.0)

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="api_test.wav",
        storage_path=str(audio_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        duration_seconds=1.0,
    )
    db_session.add(asset)
    db_session.commit()

    # Prevent background execution in HTTP test
    monkeypatch.setattr(
        "app.api.endpoints.transcription.process_transcription_job_sync",
        lambda *args, **kwargs: None
    )

    # 1. Start transcription
    resp = client.post(
        f"/api/v1/transcription/start/{meeting.id}",
        json={"language": "hi"},
        headers=headers,
    )
    assert resp.status_code == 202
    job_data = resp.json()
    assert job_data["status"] == JobStatus.PENDING.value
    job_id = job_data["job_id"]

    # 2. Query job status
    job_resp = client.get(f"/api/v1/transcription/jobs/{job_id}", headers=headers)
    assert job_resp.status_code == 200
    assert job_resp.json()["job_id"] == job_id

    # 3. Add a completed transcript in DB and retrieve via API
    t = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.ORIGINAL.value,
        version=1,
        language_detected="hi",
        is_primary=True,
    )
    db_session.add(t)
    db_session.flush()

    seg = TranscriptSegment(
        transcript_id=t.id,
        sequence_order=0,
        start_time=0.0,
        end_time=1.0,
        text="Aabhar",
        language_code="gu",
    )
    db_session.add(seg)
    db_session.commit()

    list_resp = client.get(f"/api/v1/transcription/{meeting.id}", headers=headers)
    assert list_resp.status_code == 200
    transcripts = list_resp.json()
    assert len(transcripts) == 1
    assert transcripts[0]["version"] == 1
    assert len(transcripts[0]["segments"]) == 1
    assert transcripts[0]["segments"][0]["text"] == "Aabhar"
    assert transcripts[0]["segments"][0]["language_code"] == "gu"


# --- 5. Phase 4 Verification & Quality Assurance Tests ---

def test_speech_crossing_chunk_boundary():
    """Verify timestamp math and seamless deduplication when speech crosses a chunk boundary.
    
    Scenario:
    Chunk 0: [0.0s - 300.0s]
    Chunk 1: [298.0s - 600.0s] (2.0s overlap)
    
    A sentence begins at 298.5s and ends at 301.5s:
    - Chunk 0 was cut off at 300.0s, yielding partial fragment 'Let us proceed with the' [298.5s - 300.0s].
    - Chunk 1 starts at 298.0s and captures the complete utterance:
      Local: [0.5s - 3.5s] -> Absolute: [298.5s - 301.5s] ('Let us proceed with the system architecture review').
    - Followed by next sentence:
      Local: [4.0s - 8.0s] -> Absolute: [302.0s - 306.0s] ('Next agenda item is security').
    """
    from app.services.transcription.chunking import AudioChunk, merge_and_deduplicate_segments

    chunk0 = AudioChunk(index=0, start_time=0.0, end_time=300.0, is_last=False)
    chunk1 = AudioChunk(index=1, start_time=298.0, end_time=600.0, is_last=True)

    segs0 = [
        ASRSegment(
            sequence_order=0,
            start_time=290.0,
            end_time=295.0,
            text="Welcome to the meeting.",
            language_code="en"
        ),
        ASRSegment(
            sequence_order=1,
            start_time=298.5,
            end_time=300.0,
            text="Let us proceed with the",  # Boundary cut-off fragment
            language_code="en"
        ),
    ]

    segs1 = [
        ASRSegment(
            sequence_order=0,
            start_time=0.5,
            end_time=3.5,
            text="Let us proceed with the system architecture review",  # Complete utterance
            language_code="en"
        ),
        ASRSegment(
            sequence_order=1,
            start_time=4.0,
            end_time=8.0,
            text="Next agenda item is security",
            language_code="en"
        ),
    ]

    merged = merge_and_deduplicate_segments([(chunk0, segs0), (chunk1, segs1)], overlap=2.0)

    # Must contain 3 segments, not 4 (the partial cut-off fragment must be replaced, not repeated)
    assert len(merged) == 3
    assert merged[0].text == "Welcome to the meeting."
    assert merged[0].start_time == 290.0
    assert merged[0].end_time == 295.0

    # Boundary crossing segment: complete text, adjusted offset
    assert merged[1].text == "Let us proceed with the system architecture review"
    assert merged[1].start_time == 298.5
    assert merged[1].end_time == 301.5

    # Subsequent segment
    assert merged[2].text == "Next agenda item is security"
    assert merged[2].start_time == 302.0
    assert merged[2].end_time == 306.0

    # Ensure strictly consecutive sequence ordering
    assert [m.sequence_order for m in merged] == [0, 1, 2]


def test_hinglish_mode_prompt_conditioning_and_genuine_metadata():
    """Verify Hinglish mode uses decoder prompt conditioning without fabricating language codes."""
    engine = FasterWhisperEngine()

    mock_whisper = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "hi"  # Genuine model detection is Hindi
    mock_info.language_probability = 0.88

    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = "Yeh project bahut important hai, let's deploy today."
    mock_segment.avg_logprob = -0.22

    mock_whisper.transcribe.return_value = ([mock_segment], mock_info)
    engine.model = mock_whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _create_synthetic_16k_mono_wav(f.name, duration_sec=2.0)
        temp_wav = f.name

    try:
        # Request Hinglish mode (or legacy 'hi-en')
        opts = TranscriptionOptions(transcription_mode="hinglish")
        result = engine.transcribe(temp_wav, options=opts)

        # 1. Assert whisper transcribe was called with language=None (auto multilingual)
        call_kwargs = mock_whisper.transcribe.call_args.kwargs
        assert call_kwargs["language"] is None, "Whisper must not be passed 'hi-en' as language"
        assert "Hindi and English" in call_kwargs["initial_prompt"], "Initial prompt must condition decoder"

        # 2. Assert language metadata is NOT fabricated as 'hi-en'
        assert result.detected_language == "hi", "Detected language must be genuine model detection ('hi'), not 'hi-en'"
        assert result.transcription_mode == "hinglish"
        assert result.segments[0].language_code == "hi"

        # 3. Assert confidence score is bounded [0.0, 1.0] and distinguished from avg_logprob
        assert result.segments[0].avg_logprob == -0.22
        assert 0.0 <= result.segments[0].confidence <= 1.0
        import math
        assert abs(result.segments[0].confidence - round(math.exp(-0.22), 4)) < 0.001
    finally:
        if os.path.isfile(temp_wav):
            os.remove(temp_wav)


def test_confidence_metadata_logprob_distinction():
    """Verify confidence is bounded [0.0, 1.0] and properly distinguished from model log probabilities."""
    import math

    engine = FasterWhisperEngine()
    mock_whisper = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99

    # Test segment with negative log-probability
    mock_seg = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end = 1.0
    mock_seg.text = "Testing confidence scores."
    mock_seg.avg_logprob = -0.5

    mock_whisper.transcribe.return_value = ([mock_seg], mock_info)
    engine.model = mock_whisper

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _create_synthetic_16k_mono_wav(f.name, duration_sec=1.0)
        temp_wav = f.name

    try:
        result = engine.transcribe(temp_wav)
        seg = result.segments[0]
        # avg_logprob is negative model token log-probability
        assert seg.avg_logprob == -0.5
        # confidence is an uncalibrated heuristic score bounded [0.0, 1.0]
        assert seg.confidence is not None
        assert 0.0 <= seg.confidence <= 1.0
        assert seg.confidence == round(math.exp(-0.5), 4)
    finally:
        if os.path.isfile(temp_wav):
            os.remove(temp_wav)


def test_pyav_compatibility_shim_safe_behavior():
    """Verify apply_pyav_compatibility_shim intercepts metadata_errors without breaking open behavior."""
    import av
    from app.services.transcription.gpu_manager import apply_pyav_compatibility_shim

    apply_pyav_compatibility_shim()

    # av.open should have the shim attribute
    assert hasattr(av.open, "_is_compat_shim")
    assert av.open._is_compat_shim is True

    # Test that calling av.open with metadata_errors="ignore" does not raise TypeError
    # We test on a dummy file or verify keyword removal
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _create_synthetic_16k_mono_wav(f.name, duration_sec=0.1)
        test_file = f.name

    try:
        container = av.open(test_file, mode="r", metadata_errors="ignore")
        assert container is not None
        container.close()
    finally:
        if os.path.isfile(test_file):
            os.remove(test_file)

