import io
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from app.models.identity import Organization, User
from app.models.meeting import Meeting, MediaAsset, ProcessingJob, JobStatus, JobType
from app.models.memory import AuditLog
from app.services.media_validation import (
    validate_file_extension,
    validate_media_content,
    validate_youtube_url,
    MediaValidationError,
)
from app.services.media_storage import (
    get_meeting_storage_dir,
    generate_asset_filename,
    compute_sha256,
    save_upload_stream,
    to_safe_relative_path,
    resolve_safe_storage_path,
    StorageError,
)
from app.services.media_metadata import extract_media_metadata
from app.services.audio_extraction import normalize_audio_for_asr
from app.services.youtube_ingestion import extract_youtube_stream, YouTubeIngestionError
from app.services.job_manager import create_processing_job, update_processing_job
from app.services.lifecycle import delete_raw_media_asset, MediaLifecycleError


@pytest.fixture(scope="module")
def synthetic_media(tmp_path_factory):
    """Generate minimal real synthetic media files using FFmpeg for testing."""
    temp_dir = tmp_path_factory.mktemp("test_media")
    wav_path = temp_dir / "test_audio.wav"
    mp4_path = temp_dir / "test_video.mp4"

    # 1. Synthetic 1-second 44.1kHz stereo audio
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=1:sample_rate=44100",
            "-ac", "2",
            str(wav_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    # 2. Synthetic 1-second video with audio track
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
            "-c:v", "libx264", "-c:a", "aac",
            str(mp4_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return {"wav": wav_path, "mp4": mp4_path, "dir": temp_dir}


def test_supported_and_unsupported_extensions():
    """Verify whitelist validation for audio/video formats and rejection of unsupported files."""
    valid_audio = ["recording.wav", "meeting.MP3", "audio.m4a", "voice.FLAC"]
    for filename in valid_audio:
        ext, category = validate_file_extension(filename)
        assert category == "AUDIO"
        assert ext in {"wav", "mp3", "m4a", "flac"}

    valid_video = ["presentation.mp4", "zoom.mkv", "recording.MOV", "stream.webm"]
    for filename in valid_video:
        ext, category = validate_file_extension(filename)
        assert category == "VIDEO"
        assert ext in {"mp4", "mkv", "mov", "webm"}

    # Unsupported extensions
    invalid_files = ["malicious.exe", "script.sh", "notes.pdf", "image.png", "no_ext"]
    for filename in invalid_files:
        with pytest.raises(MediaValidationError):
            validate_file_extension(filename)


def test_youtube_url_validation():
    """Verify validation of YouTube URLs and rejection of invalid/unsupported domains."""
    valid_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtube.com/watch?v=dQw4w9WgXcQ&feature=shared",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    ]
    for url in valid_urls:
        is_valid, video_id, err = validate_youtube_url(url)
        assert is_valid is True, f"Failed for {url}"
        assert video_id == "dQw4w9WgXcQ"
        assert err is None

    # Invalid / Unsupported
    invalid_urls = [
        ("https://vimeo.com/123456789", "Unsupported domain"),
        ("https://dailymotion.com/video/x123", "Unsupported domain"),
        ("https://malicious.org/watch?v=123", "Unsupported domain"),
        ("ftp://youtube.com/watch?v=12345678901", "Invalid URL scheme"),
        ("not_a_valid_url", "Invalid URL scheme"),
    ]
    for url, reason in invalid_urls:
        is_valid, video_id, err = validate_youtube_url(url)
        assert is_valid is False
        assert video_id is None
        assert err is not None


def test_metadata_extraction_and_checksum(synthetic_media):
    """Verify FFprobe metadata extraction and SHA-256 calculation on real media."""
    wav_path = synthetic_media["wav"]
    metadata = extract_media_metadata(wav_path)
    assert metadata.has_audio is True
    assert metadata.channels == 2
    assert metadata.sample_rate == 44100
    assert metadata.duration_seconds is not None
    assert metadata.duration_seconds > 0.9

    checksum = compute_sha256(wav_path)
    assert len(checksum) == 64
    assert checksum == compute_sha256(wav_path)


def test_audio_normalization_16k_mono(synthetic_media, tmp_path):
    """Verify audio extraction from video and normalization to 16kHz mono WAV for future ASR."""
    mp4_path = synthetic_media["mp4"]
    output_wav = tmp_path / "normalized_out.wav"

    norm_path, norm_meta, checksum = normalize_audio_for_asr(mp4_path, output_wav)

    assert norm_path.exists()
    assert norm_meta.has_audio is True
    assert norm_meta.sample_rate == 16000     # 16 kHz ASR requirement
    assert norm_meta.channels == 1           # Mono channel ASR requirement
    assert norm_meta.audio_codec == "pcm_s16le"
    assert len(checksum) == 64


def test_file_size_validation_limit(tmp_path):
    """Verify that uploading content exceeding the size limit is aborted and rejected."""
    dest_path = tmp_path / "test_limit.bin"
    # Create stream with 1MB data, set max to 500KB
    data = b"X" * (1024 * 1024)
    stream = io.BytesIO(data)

    with pytest.raises(StorageError) as exc_info:
        save_upload_stream(stream, dest_path, max_bytes=500 * 1024)

    assert "exceeds maximum permitted upload limit" in str(exc_info.value)
    assert not dest_path.exists()


def test_secure_storage_path_traversal_prevention(tmp_path):
    """Verify path traversal attacks are rejected and paths are resolved safely."""
    meeting_id = str(uuid.uuid4())
    meeting_dir = get_meeting_storage_dir(meeting_id)
    assert meeting_dir.exists()

    # Relative path conversion
    test_file = meeting_dir / "sample.wav"
    test_file.touch()
    safe_rel = to_safe_relative_path(test_file)
    assert not safe_rel.startswith("C:")
    assert not safe_rel.startswith("/")
    assert safe_rel.startswith("storage/")

    resolved = resolve_safe_storage_path(safe_rel)
    assert resolved == test_file.resolve()

    # Malicious traversal injection
    with pytest.raises(StorageError):
        resolve_safe_storage_path("storage/../../windows/system32/cmd.exe")


def test_processing_job_lifecycle(db_session):
    """Verify job creation, progress tracking, and status transitions."""
    org = Organization(name="Job Org", slug="job-org")
    meeting = Meeting(organization_id=org.id, title="Job Meeting")
    db_session.add_all([org, meeting])
    db_session.commit()

    job = create_processing_job(db_session, meeting.id, JobType.INGESTION.value)
    assert job.status == JobStatus.PENDING.value
    assert job.progress == 0.0

    update_processing_job(db_session, job.id, status=JobStatus.IN_PROGRESS.value, progress=50.0)
    reloaded = db_session.get(ProcessingJob, job.id)
    assert reloaded.status == JobStatus.IN_PROGRESS.value
    assert reloaded.progress == 50.0

    update_processing_job(db_session, job.id, status=JobStatus.COMPLETED.value, progress=100.0)
    completed = db_session.get(ProcessingJob, job.id)
    assert completed.status == JobStatus.COMPLETED.value
    assert completed.progress == 100.0


def test_safe_deletion_and_audit_logging(db_session, tmp_path):
    """Verify safe deletion of raw media file, DB flag update, and AuditLog recording."""
    org = Organization(name="Lifecycle Org", slug="lifecycle-org")
    user = User(email="privacy@corp.com", full_name="Privacy Officer")
    meeting = Meeting(organization_id=org.id, title="Secret Meeting")
    db_session.add_all([org, user, meeting])
    db_session.commit()

    # Create dummy raw media asset in meeting folder
    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_file = meeting_dir / "raw_audio.wav"
    raw_file.write_bytes(b"RIFF dummy audio data")

    asset = MediaAsset(
        meeting_id=meeting.id,
        file_name="raw_audio.wav",
        storage_path=to_safe_relative_path(raw_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        file_size_bytes=raw_file.stat().st_size,
        is_raw_deleted=False
    )
    db_session.add(asset)
    db_session.commit()

    # Perform safe deletion
    deleted_asset = delete_raw_media_asset(db_session, asset.id, user_id=user.id)
    assert deleted_asset.is_raw_deleted is True
    assert not raw_file.exists()

    # Check AuditLog
    audits = db_session.query(AuditLog).filter(AuditLog.entity_id == asset.id).all()
    assert len(audits) == 1
    assert audits[0].action == "DELETE_RAW_AUDIO"
    assert audits[0].user_id == user.id


from backend.tests.conftest import make_auth_headers


def test_api_upload_endpoint(client, synthetic_media, db_session, monkeypatch, org_admin_headers):
    """Verify POST /api/v1/media/upload creates meeting, job, asset, and accepts upload."""
    # Patch background task so test is non-blocking and isolated
    monkeypatch.setattr("app.api.endpoints.media._process_uploaded_media_async", lambda *args, **kwargs: None)

    wav_path = synthetic_media["wav"]
    with open(wav_path, "rb") as f:
        response = client.post(
            "/api/v1/media/upload",
            files={"file": ("interview.wav", f, "audio/wav")},
            data={"title": "Q3 Executive Interview", "description": "Candidate interview audio."},
            headers=org_admin_headers,
        )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert data["meeting_id"] is not None
    assert data["job_id"] is not None
    assert data["media_asset_id"] is not None

    # Check that meeting exists in DB
    meeting = db_session.get(Meeting, data["meeting_id"])
    assert meeting is not None
    assert meeting.title == "Q3 Executive Interview"


def test_api_youtube_endpoint_authorization_check(client, monkeypatch, org_admin_headers):
    """Verify POST /api/v1/media/youtube requires explicit user authorization confirmation."""
    # 1. Unconfirmed authorization should be rejected
    res_rejected = client.post(
        "/api/v1/media/youtube",
        json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "confirmed_authorization": False
        },
        headers=org_admin_headers,
    )
    assert res_rejected.status_code == 400
    assert "authorization" in res_rejected.json()["detail"].lower()

    # 2. Confirmed authorization accepted
    monkeypatch.setattr("app.api.endpoints.media._process_youtube_media_async", lambda *args, **kwargs: None)
    res_accepted = client.post(
        "/api/v1/media/youtube",
        json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Public Keynote",
            "confirmed_authorization": True
        },
        headers=org_admin_headers,
    )
    assert res_accepted.status_code == 202
    data = res_accepted.json()
    assert data["status"] == "accepted"
    assert data["meeting_id"] is not None
    assert data["job_id"] is not None


def test_api_organization_isolation(client, db_session):
    """Verify tenant isolation: requests cannot access another organization's meetings/media."""
    org_a = Organization(name="Org Alpha", slug="org-alpha", is_active=True)
    org_b = Organization(name="Org Beta", slug="org-beta", is_active=True)
    db_session.add_all([org_a, org_b])
    db_session.flush()

    from app.models.identity import User, OrganizationMembership, OrgRole
    from app.core.security import hash_password
    user_a = User(
        email="usera@orgalpha.com",
        full_name="User A",
        hashed_password=hash_password("Pass12345!"),
        is_active=True,
    )
    db_session.add(user_a)
    db_session.flush()

    m_a = OrganizationMembership(
        organization_id=org_a.id,
        user_id=user_a.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m_a)

    meeting_b = Meeting(organization_id=org_b.id, title="Beta Confidential")
    db_session.add(meeting_b)
    db_session.commit()

    headers_a = make_auth_headers(user_a, org_a)

    # User in Org A attempts to access Org B meeting
    response = client.get(
        f"/api/v1/media/meetings/{meeting_b.id}",
        headers=headers_a,
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Meeting not found."


def test_process_youtube_media_async_completes_job_and_creates_assets(db_session, tmp_path, monkeypatch):
    """Verify that _process_youtube_media_async calls extract_youtube_stream with progress_hook and completes."""
    from unittest.mock import patch
    from app.api.endpoints.media import _process_youtube_media_async
    from app.models.identity import Organization
    from app.models.meeting import Meeting, ProcessingJob, MediaAsset, JobStatus, JobType
    from app.services.media_metadata import MediaMetadata

    monkeypatch.setattr("app.api.endpoints.media.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "close", lambda: None)

    org = Organization(name="Test Ingest Org", slug=f"test-ingest-org-{uuid.uuid4().hex[:6]}")
    db_session.add(org)
    db_session.commit()

    # Setup DB records
    meeting = Meeting(organization_id=org.id, title="YouTube Video test123", status="INGESTING")
    db_session.add(meeting)
    db_session.commit()

    job = ProcessingJob(meeting_id=meeting.id, job_type=JobType.INGESTION.value, status=JobStatus.PENDING.value)
    db_session.add(job)
    db_session.commit()

    raw_file = tmp_path / "yt_test_raw.webm"
    raw_file.write_bytes(b"dummy raw audio bytes")
    norm_file = tmp_path / "yt_test_normalized.wav"
    norm_file.write_bytes(b"dummy normalized wav bytes")

    dummy_meta = MediaMetadata(
        format_name="wav",
        duration_seconds=42.0,
        sample_rate=16000,
        channels=1,
        audio_codec="pcm_s16le",
        file_size_bytes=norm_file.stat().st_size
    )
    raw_dummy_meta = MediaMetadata(
        format_name="webm",
        duration_seconds=42.0,
        sample_rate=48000,
        channels=2,
        audio_codec="opus",
        file_size_bytes=raw_file.stat().st_size
    )

    def mock_extract(youtube_url, output_dir, confirmed_authorization=False, raw_asset_id=None, progress_hook=None):
        if progress_hook:
            progress_hook(50.0)
        return (
            raw_file,
            norm_file,
            {"id": "test123", "title": "Actual YouTube Title", "description": "Actual Description"},
            dummy_meta,
            "dummy_checksum_123"
        )

    with patch("app.api.endpoints.media.extract_youtube_stream", side_effect=mock_extract), \
         patch("app.api.endpoints.media.extract_media_metadata", return_value=raw_dummy_meta):
        _process_youtube_media_async(
            meeting_id=meeting.id,
            job_id=job.id,
            youtube_url="https://www.youtube.com/watch?v=test123",
            meeting_dir_str=str(tmp_path),
            raw_asset_id="asset-yt-raw-1",
        )

    db_session.refresh(job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.progress == 100.0

    db_session.refresh(meeting)
    assert meeting.title == "Actual YouTube Title"

    assets = db_session.query(MediaAsset).filter(MediaAsset.meeting_id == meeting.id).all()
    assert len(assets) == 2
