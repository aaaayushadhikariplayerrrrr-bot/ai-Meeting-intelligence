import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    BackgroundTasks,
    status
)
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api.deps import get_current_tenant_membership, require_org_admin, TenantContext
from app.core.config import settings
from app.database.session import get_db, SessionLocal
from app.models.identity import Organization
from app.models.meeting import (
    Meeting,
    MediaAsset,
    ProcessingJob,
    MeetingStatus,
    JobStatus,
    JobType,
    InputSourceType,
)
from app.schemas.media import (
    MediaAssetResponse,
    ProcessingJobResponse,
    MeetingResponse,
    YouTubeIngestRequest,
    IngestResponse,
)
from app.services.media_validation import (
    validate_file_extension,
    validate_media_content,
    validate_media_duration,
    validate_youtube_url,
    MediaValidationError,
)
from app.services.media_storage import (
    get_meeting_storage_dir,
    generate_asset_filename,
    save_upload_stream,
    to_safe_relative_path,
    compute_sha256,
    StorageError,
)
from app.services.media_metadata import extract_media_metadata, MetadataExtractionError
from app.services.audio_extraction import normalize_audio_for_asr
from app.services.youtube_ingestion import extract_youtube_stream, YouTubeIngestionError
from app.services.job_manager import create_processing_job, update_processing_job
from app.services.lifecycle import delete_raw_media_asset, MediaLifecycleError
from app.services.meeting_lifecycle import reconcile_meeting_status

router = APIRouter()


def _process_uploaded_media_async(
    meeting_id: str,
    job_id: str,
    raw_asset_id: str,
    raw_file_path_str: str,
    normalized_filename: str
):
    """Background worker task to extract metadata, inspect content, and normalize audio for ASR."""
    db: Session = SessionLocal()
    try:
        raw_path = Path(raw_file_path_str)
        if not raw_path.exists():
            update_processing_job(
                db,
                job_id,
                status=JobStatus.FAILED.value,
                error_message="Raw media file does not exist on disk.",
                completed_at=datetime.now(timezone.utc),
            )
            meeting = db.get(Meeting, meeting_id)
            if meeting:
                reconcile_meeting_status(db, meeting_id)
            return

        update_processing_job(db, job_id, status=JobStatus.IN_PROGRESS.value, progress=15.0)

        # 1. Content validation via ffprobe
        is_valid, err = validate_media_content(raw_path)
        if not is_valid:
            update_processing_job(
                db,
                job_id,
                status=JobStatus.FAILED.value,
                error_message=f"Media validation failed: {err}",
                completed_at=datetime.now(timezone.utc),
            )
            meeting = db.get(Meeting, meeting_id)
            if meeting:
                reconcile_meeting_status(db, meeting_id)
            return

        update_processing_job(db, job_id, progress=35.0)

        # 2. Extract metadata for original media and re-verify duration limit
        raw_metadata = extract_media_metadata(raw_path)
        try:
            validated_dur = validate_media_duration(raw_metadata.duration_seconds)
        except MediaValidationError as e:
            update_processing_job(
                db,
                job_id,
                status=JobStatus.FAILED.value,
                error_message=f"Media duration validation failed: {str(e)}",
                completed_at=datetime.now(timezone.utc),
            )
            meeting = db.get(Meeting, meeting_id)
            if meeting:
                reconcile_meeting_status(db, meeting_id)
            return

        raw_asset = db.get(MediaAsset, raw_asset_id)
        if raw_asset:
            raw_asset.duration_seconds = validated_dur
            raw_asset.file_size_bytes = raw_path.stat().st_size
            db.commit()

        update_processing_job(db, job_id, progress=55.0)

        # 3. Normalize audio track to standard 16kHz mono WAV for future Phase 4 ASR
        normalized_output_path = raw_path.parent / normalized_filename
        norm_path, norm_meta, norm_checksum = normalize_audio_for_asr(
            raw_path,
            normalized_output_path
        )

        update_processing_job(db, job_id, progress=85.0)

        # 4. Register or update normalized audio asset (idempotency protection)
        existing_norm = (
            db.query(MediaAsset)
            .filter(
                MediaAsset.meeting_id == meeting_id,
                MediaAsset.file_name == normalized_filename,
                MediaAsset.media_type == "NORMALIZED_AUDIO",
            )
            .first()
        )
        if existing_norm:
            existing_norm.duration_seconds = norm_meta.duration_seconds
            existing_norm.file_size_bytes = norm_path.stat().st_size
            existing_norm.checksum_sha256 = norm_checksum
            existing_norm.is_raw_deleted = False
        else:
            normalized_asset = MediaAsset(
                meeting_id=meeting_id,
                file_name=normalized_filename,
                storage_path=to_safe_relative_path(norm_path),
                media_type="NORMALIZED_AUDIO",
                mime_type="audio/wav",
                file_size_bytes=norm_path.stat().st_size,
                duration_seconds=norm_meta.duration_seconds,
                checksum_sha256=norm_checksum,
                is_raw_deleted=False
            )
            db.add(normalized_asset)

        # 5. Complete processing — let reconciler set COMPLETED; commit normalized asset first
        db.commit()
        update_processing_job(
            db,
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=100.0,
            completed_at=datetime.now(timezone.utc),
        )
        # reconcile_meeting_status is called by update_processing_job on COMPLETED
    except Exception as e:
        update_processing_job(
            db,
            job_id,
            status=JobStatus.FAILED.value,
            error_message=f"Audio extraction failed: {str(e)}",
            completed_at=datetime.now(timezone.utc),
        )
        # reconcile_meeting_status is called by update_processing_job on FAILED
    finally:
        db.close()


def _process_youtube_media_async(
    meeting_id: str,
    job_id: str,
    youtube_url: str,
    meeting_dir_str: str,
    raw_asset_id: Optional[str] = None
):
    """Background worker task to extract and normalize YouTube stream."""
    db: Session = SessionLocal()
    try:
        update_processing_job(db, job_id, status=JobStatus.IN_PROGRESS.value, progress=15.0)
        output_dir = Path(meeting_dir_str)

        if not raw_asset_id:
            raw_asset_id = str(uuid.uuid4())

        raw_media_path, norm_wav_path, info, norm_meta, checksum = extract_youtube_stream(
            youtube_url=youtube_url,
            output_dir=output_dir,
            confirmed_authorization=True,
            raw_asset_id=raw_asset_id
        )

        update_processing_job(db, job_id, progress=65.0)

        meeting = db.get(Meeting, meeting_id)
        if meeting:
            if not meeting.title or meeting.title == "YouTube Meeting Ingestion":
                meeting.title = info.get("title", meeting.title)
            if not meeting.description:
                meeting.description = info.get("description", "")
            db.commit()

        # Save raw asset
        raw_asset = MediaAsset(
            id=raw_asset_id,
            meeting_id=meeting_id,
            file_name=raw_media_path.name,
            storage_path=to_safe_relative_path(raw_media_path),
            media_type="YOUTUBE_RAW",
            mime_type="audio/mp4",
            file_size_bytes=raw_media_path.stat().st_size if raw_media_path.exists() else None,
            duration_seconds=float(info.get("duration") or 0.0),
            checksum_sha256=compute_sha256(raw_media_path) if raw_media_path.exists() else None,
            is_raw_deleted=False
        )
        db.add(raw_asset)
        db.flush()

        # Save normalized 16kHz mono WAV asset for future ASR (idempotency check)
        existing_norm = (
            db.query(MediaAsset)
            .filter(
                MediaAsset.meeting_id == meeting_id,
                MediaAsset.file_name == norm_wav_path.name,
                MediaAsset.media_type == "NORMALIZED_AUDIO",
            )
            .first()
        )
        if existing_norm:
            existing_norm.duration_seconds = norm_meta.duration_seconds
            existing_norm.file_size_bytes = norm_wav_path.stat().st_size
            existing_norm.checksum_sha256 = checksum
            existing_norm.is_raw_deleted = False
        else:
            normalized_asset = MediaAsset(
                meeting_id=meeting_id,
                file_name=norm_wav_path.name,
                storage_path=to_safe_relative_path(norm_wav_path),
                media_type="NORMALIZED_AUDIO",
                mime_type="audio/wav",
                file_size_bytes=norm_wav_path.stat().st_size,
                duration_seconds=norm_meta.duration_seconds,
                checksum_sha256=checksum,
                is_raw_deleted=False
            )
            db.add(normalized_asset)
        db.commit()
        update_processing_job(
            db,
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=100.0,
            completed_at=datetime.now(timezone.utc),
        )
        # reconcile_meeting_status is called by update_processing_job on COMPLETED
    except Exception as e:
        update_processing_job(
            db,
            job_id,
            status=JobStatus.FAILED.value,
            error_message=f"YouTube ingestion failed: {str(e)}",
            completed_at=datetime.now(timezone.utc),
        )
        # reconcile_meeting_status is called by update_processing_job on FAILED
        # Clean up any partial files or downloaded streams on failure
        output_dir = Path(meeting_dir_str)
        if output_dir.exists():
            for f in output_dir.glob("*"):
                try:
                    if f.is_file():
                        f.unlink(missing_ok=True)
                except Exception:
                    pass
    finally:
        db.close()


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Media File",
    description="Upload an audio or video file for meeting ingestion. Dispatches background extraction."
)
def upload_media_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> IngestResponse:
    # 1. Extension and media type check
    try:
        ext, media_category = validate_file_extension(file.filename or "")
    except MediaValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    org = tenant.organization
    user_id = tenant.user.id

    # 2. Pre-generate meeting ID and storage directory to stream upload before DB operations
    meeting_id = str(uuid.uuid4())
    meeting_dir = get_meeting_storage_dir(meeting_id)
    raw_storage_filename = generate_asset_filename(ext)
    dest_path = meeting_dir / raw_storage_filename

    # 3. Stream upload to secure storage directory
    try:
        bytes_written, sha256_checksum = save_upload_stream(
            file.file,
            dest_path,
            max_bytes=settings.MAX_UPLOAD_SIZE_BYTES
        )
    except StorageError as e:
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        if meeting_dir.exists() and not any(meeting_dir.iterdir()):
            meeting_dir.rmdir()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 4. Enforce FFprobe content and duration validation BEFORE creating DB entities or jobs
    try:
        is_valid, err = validate_media_content(dest_path)
        if not is_valid:
            raise MediaValidationError(f"Media validation failed: {err}")

        raw_metadata = extract_media_metadata(dest_path)
        validated_duration = validate_media_duration(raw_metadata.duration_seconds)
    except (MediaValidationError, MetadataExtractionError) as e:
        # Clean up uploaded file and directory so no orphaned files remain on disk
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        if meeting_dir.exists() and not any(meeting_dir.iterdir()):
            meeting_dir.rmdir()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 5. Validation passed: Create Meeting entity
    meeting_title = (title or "").strip() or Path(file.filename or "Meeting").stem
    meeting = Meeting(
        id=meeting_id,
        organization_id=org.id,
        created_by_user_id=user_id,
        title=meeting_title,
        description=description,
        input_source_type=InputSourceType.AUDIO.value if media_category == "AUDIO" else InputSourceType.VIDEO.value,
        status=MeetingStatus.INGESTING.value
    )
    db.add(meeting)

    raw_asset_id = str(uuid.uuid4())
    normalized_filename = f"{raw_asset_id}_normalized_16k.wav"

    # 6. Create persistent background ProcessingJob with execution payload
    job_payload = {
        "subtype": "UPLOAD",
        "raw_asset_id": raw_asset_id,
        "raw_file_path": str(dest_path),
        "normalized_filename": normalized_filename,
    }
    job = create_processing_job(
        db,
        meeting_id=meeting.id,
        job_type=JobType.INGESTION.value,
        initial_status=JobStatus.PENDING.value,
        payload=job_payload,
    )

    # 7. Create raw media asset record with verified duration
    raw_asset = MediaAsset(
        id=raw_asset_id,
        meeting_id=meeting.id,
        file_name=file.filename or raw_storage_filename,
        storage_path=to_safe_relative_path(dest_path),
        media_type=media_category,
        mime_type=file.content_type or f"application/{ext}",
        file_size_bytes=bytes_written,
        duration_seconds=validated_duration,
        checksum_sha256=sha256_checksum,
        is_raw_deleted=False
    )
    db.add(raw_asset)
    db.commit()
    db.refresh(raw_asset)

    return IngestResponse(
        meeting_id=meeting.id,
        job_id=job.id,
        status="accepted",
        message="Media uploaded successfully. Background audio normalization queued.",
        media_asset_id=raw_asset.id
    )


@router.post(
    "/youtube",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest YouTube Meeting",
    description="Ingest a meeting from a user-provided YouTube URL. Requires authorization confirmation."
)
def ingest_youtube_meeting(
    payload: YouTubeIngestRequest,
    background_tasks: BackgroundTasks,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> IngestResponse:
    if not payload.confirmed_authorization:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must confirm you have authorization to process this YouTube content."
        )

    is_valid, video_id, err = validate_youtube_url(payload.url)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)

    org = tenant.organization
    user_id = tenant.user.id

    meeting_title = (payload.title or "").strip() or f"YouTube Video {video_id}"
    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user_id,
        title=meeting_title,
        description=payload.description,
        input_source_type=InputSourceType.YOUTUBE.value,
        source_url_or_reference=payload.url,
        status=MeetingStatus.INGESTING.value
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)

    meeting_dir = get_meeting_storage_dir(meeting.id)
    raw_asset_id = str(uuid.uuid4())

    job_payload = {
        "subtype": "YOUTUBE",
        "youtube_url": payload.url,
        "meeting_dir": str(meeting_dir),
        "raw_asset_id": raw_asset_id,
    }
    job = create_processing_job(
        db,
        meeting_id=meeting.id,
        job_type=JobType.INGESTION.value,
        initial_status=JobStatus.PENDING.value,
        payload=job_payload,
    )

    return IngestResponse(
        meeting_id=meeting.id,
        job_id=job.id,
        status="accepted",
        message="YouTube ingestion job accepted and queued in background."
    )


@router.get(
    "/meetings",
    response_model=List[MeetingResponse],
    summary="List Organization Meetings",
    description="List all meetings with associated media assets and processing jobs."
)
def list_meetings(
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> List[MeetingResponse]:
    stmt = (
        select(Meeting)
        .where(Meeting.organization_id == tenant.organization.id)
        .order_by(Meeting.created_at.desc())
    )
    meetings = db.execute(stmt).scalars().all()
    return meetings


@router.get(
    "/meetings/{meeting_id}",
    response_model=MeetingResponse,
    summary="Get Meeting Details",
    description="Get specific meeting details with media assets and jobs."
)
def get_meeting(
    meeting_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> MeetingResponse:
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting not found."
        )
    return meeting


@router.get(
    "/jobs/{job_id}",
    response_model=ProcessingJobResponse,
    summary="Get Processing Job Status",
    description="Poll progress and status of a media ingestion/processing job."
)
def get_job_status(
    job_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> ProcessingJobResponse:
    stmt = (
        select(ProcessingJob)
        .join(Meeting, ProcessingJob.meeting_id == Meeting.id)
        .where(
            ProcessingJob.id == job_id,
            Meeting.organization_id == tenant.organization.id,
        )
    )
    job = db.execute(stmt).scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found."
        )

    return job


@router.delete(
    "/assets/{media_asset_id}",
    response_model=MediaAssetResponse,
    summary="Safely Delete Raw Media",
    description="Safely removes physical media file from disk and records an audit log. Requires ORG_ADMIN."
)
def delete_media_asset(
    media_asset_id: str,
    tenant: TenantContext = Depends(require_org_admin),
    db: Session = Depends(get_db)
) -> MediaAssetResponse:
    stmt = (
        select(MediaAsset)
        .join(Meeting, MediaAsset.meeting_id == Meeting.id)
        .where(
            MediaAsset.id == media_asset_id,
            Meeting.organization_id == tenant.organization.id,
        )
    )
    asset = db.execute(stmt).scalar_one_or_none()
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media asset not found."
        )

    try:
        updated_asset = delete_raw_media_asset(db, media_asset_id, user_id=tenant.user.id)
        return updated_asset
    except MediaLifecycleError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
