import io
import math
import uuid
import json
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    BackgroundTasks,
    Query,
    Request,
    status
)
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_, and_

from app.api.deps import get_current_tenant_membership, require_org_admin, TenantContext
from app.core.config import settings
from app.database.session import get_db, SessionLocal
from app.models.identity import Organization
from app.models.meeting import (
    Meeting,
    MeetingParticipant,
    MediaAsset,
    ProcessingJob,
    MeetingStatus,
    JobStatus,
    JobType,
    InputSourceType,
)
from app.models.project import Project
from app.models.memory import AuditLog
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
)
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.services.report_generator import generate_meeting_pdf_report
from app.schemas.media import (
    MediaAssetResponse,
    ProcessingJobResponse,
    MeetingResponse,
    MeetingParticipantCreate,
    MeetingParticipantResponse,
    PaginatedMeetingsResponse,
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
from app.services.lifecycle import delete_raw_media_asset, resolve_safe_storage_path, MediaLifecycleError
from app.services.meeting_lifecycle import reconcile_meeting_status
from app.services.memory.storage import ChunkStorageManager

logger = logging.getLogger(__name__)

router = APIRouter()


def _process_uploaded_media_async(
    meeting_id: str,
    job_id: str,
    raw_asset_id: str,
    raw_file_path_str: str,
    normalized_filename: str,
):
    """Synchronous worker function that normalizes audio for ASR in a worker thread."""
    db = SessionLocal()
    try:
        raw_path = Path(raw_file_path_str)
        meeting_dir = raw_path.parent
        normalized_output_path = meeting_dir / normalized_filename

        update_processing_job(db, job_id, status=JobStatus.IN_PROGRESS.value, progress=10.0)

        norm_res = normalize_audio_for_asr(raw_path, normalized_output_path)
        if isinstance(norm_res, tuple):
            normalized_output_path, normalized_metadata, checksum = norm_res
        else:
            normalized_metadata = norm_res
            checksum = compute_sha256(normalized_output_path)
        update_processing_job(db, job_id, progress=75.0)

        normalized_asset = MediaAsset(
            meeting_id=meeting_id,
            file_name=normalized_filename,
            storage_path=to_safe_relative_path(normalized_output_path),
            media_type="AUDIO",
            mime_type="audio/wav",
            file_size_bytes=normalized_metadata.file_size_bytes,
            duration_seconds=normalized_metadata.duration_seconds,
            checksum_sha256=checksum,
            is_raw_deleted=False
        )
        db.add(normalized_asset)

        update_processing_job(db, job_id, status=JobStatus.COMPLETED.value, progress=100.0)
        reconcile_meeting_status(db, meeting_id)
        db.commit()

    except Exception as e:
        logger.exception(f"Background normalization failed for job {job_id}: {e}")
        try:
            update_processing_job(
                db,
                job_id,
                status=JobStatus.FAILED.value,
                error_message=f"Audio normalization failed: {str(e)}"
            )
            reconcile_meeting_status(db, meeting_id)
            db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _process_youtube_media_async(
    meeting_id: str,
    job_id: str,
    youtube_url: str,
    meeting_dir_str: str,
    raw_asset_id: str,
):
    """Synchronous worker function that downloads and normalizes YouTube audio."""
    db = SessionLocal()
    try:
        meeting_dir = Path(meeting_dir_str)
        update_processing_job(db, job_id, status=JobStatus.IN_PROGRESS.value, progress=10.0)

        def download_progress_hook(pct: float):
            job_progress = 10.0 + (pct * 0.4)
            update_processing_job(db, job_id, progress=job_progress)

        raw_media_path, normalized_path, video_info, normalized_meta, checksum = extract_youtube_stream(
            youtube_url=youtube_url,
            output_dir=meeting_dir,
            confirmed_authorization=True,
            raw_asset_id=raw_asset_id,
            progress_hook=download_progress_hook,
        )

        update_processing_job(db, job_id, progress=65.0)

        raw_meta = extract_media_metadata(raw_media_path)
        raw_checksum = compute_sha256(raw_media_path)
        raw_asset = MediaAsset(
            id=raw_asset_id,
            meeting_id=meeting_id,
            file_name=raw_media_path.name,
            storage_path=to_safe_relative_path(raw_media_path),
            media_type="AUDIO",
            mime_type=f"audio/{raw_media_path.suffix.lstrip('.')}" if raw_media_path.suffix else "audio/webm",
            file_size_bytes=raw_media_path.stat().st_size,
            duration_seconds=raw_meta.duration_seconds,
            checksum_sha256=raw_checksum,
            is_raw_deleted=False,
        )
        db.add(raw_asset)
        db.commit()

        update_processing_job(db, job_id, progress=85.0)

        normalized_asset = MediaAsset(
            meeting_id=meeting_id,
            file_name=normalized_path.name,
            storage_path=to_safe_relative_path(normalized_path),
            media_type="AUDIO",
            mime_type="audio/wav",
            file_size_bytes=normalized_meta.file_size_bytes,
            duration_seconds=normalized_meta.duration_seconds,
            checksum_sha256=checksum,
            is_raw_deleted=False,
        )
        db.add(normalized_asset)

        meeting = db.get(Meeting, meeting_id)
        if meeting:
            if not meeting.title or meeting.title.startswith("YouTube Video"):
                if video_info.get("title"):
                    meeting.title = video_info["title"]
            if not meeting.description and video_info.get("description"):
                meeting.description = video_info["description"]

        update_processing_job(db, job_id, status=JobStatus.COMPLETED.value, progress=100.0)
        reconcile_meeting_status(db, meeting_id)
        db.commit()

    except Exception as e:
        logger.exception(f"Background YouTube ingestion failed for job {job_id}: {e}")
        try:
            update_processing_job(
                db,
                job_id,
                status=JobStatus.FAILED.value,
                error_message=f"YouTube ingestion failed: {str(e)}"
            )
            reconcile_meeting_status(db, meeting_id)
            db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Media File",
    description="Upload an audio or video file for meeting ingestion with optional date, category, project, and participants."
)
def upload_media_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    meeting_date: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    participants: Optional[str] = Form(None),
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

    # Validate project belongs to organization if provided
    valid_project_id: Optional[str] = None
    if project_id and project_id.strip():
        proj_exists = db.execute(
            select(Project.id).where(
                Project.id == project_id.strip(),
                Project.organization_id == org.id
            )
        ).scalar_one_or_none()
        if not proj_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected project not found in current organization."
            )
        valid_project_id = project_id.strip()

    # Parse meeting date if provided
    actual_start: Optional[datetime] = None
    if meeting_date and meeting_date.strip():
        try:
            actual_start = datetime.fromisoformat(meeting_date.strip().replace("Z", "+00:00"))
        except Exception:
            pass

    # 2. Pre-generate meeting ID and storage directory
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
        status=MeetingStatus.INGESTING.value,
        scheduled_start_time=actual_start,
        actual_start_time=actual_start,
        category=category.strip() if category and category.strip() else None,
        project_id=valid_project_id,
    )
    db.add(meeting)

    # Process participants if provided
    if participants and participants.strip():
        try:
            p_data = json.loads(participants.strip()) if isinstance(participants, str) else participants
            if isinstance(p_data, list):
                for p_item in p_data:
                    if isinstance(p_item, dict) and p_item.get("display_name"):
                        part = MeetingParticipant(
                            meeting_id=meeting.id,
                            display_name=p_item["display_name"].strip(),
                            email=p_item.get("email", "").strip() or None,
                            role=p_item.get("role", "").strip() or None,
                            user_id=p_item.get("user_id"),
                        )
                        db.add(part)
        except Exception as e:
            logger.warning(f"Failed to parse participants during upload: {e}")

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

    # 7. Create raw media asset record
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
    description="Ingest a meeting from a user-provided YouTube URL with optional project, category, and date."
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

    valid_project_id: Optional[str] = None
    if payload.project_id and payload.project_id.strip():
        proj_exists = db.execute(
            select(Project.id).where(
                Project.id == payload.project_id.strip(),
                Project.organization_id == org.id
            )
        ).scalar_one_or_none()
        if not proj_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected project not found in current organization."
            )
        valid_project_id = payload.project_id.strip()

    actual_start: Optional[datetime] = None
    if payload.meeting_date and payload.meeting_date.strip():
        try:
            actual_start = datetime.fromisoformat(payload.meeting_date.strip().replace("Z", "+00:00"))
        except Exception:
            pass

    meeting_title = (payload.title or "").strip() or f"YouTube Video {video_id}"
    meeting = Meeting(
        organization_id=org.id,
        created_by_user_id=user_id,
        title=meeting_title,
        description=payload.description,
        input_source_type=InputSourceType.YOUTUBE.value,
        source_url_or_reference=payload.url,
        status=MeetingStatus.INGESTING.value,
        scheduled_start_time=actual_start,
        actual_start_time=actual_start,
        category=payload.category.strip() if payload.category and payload.category.strip() else None,
        project_id=valid_project_id,
    )
    db.add(meeting)
    db.flush()

    if payload.participants:
        for p_item in payload.participants:
            part = MeetingParticipant(
                meeting_id=meeting.id,
                display_name=p_item.display_name.strip(),
                email=p_item.email.strip() if p_item.email else None,
                role=p_item.role.strip() if p_item.role else None,
                user_id=p_item.user_id,
            )
            db.add(part)

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
    response_model=Union[PaginatedMeetingsResponse, List[MeetingResponse]],
    summary="List Organization Meetings",
    description="List all meetings with search, filtering, and pagination support."
)
def list_meetings(
    skip: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1, le=100),
    page: Optional[int] = Query(None, ge=1),
    search: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    participant: Optional[str] = Query(None),
    paginate: bool = Query(False),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
):
    org_id = tenant.organization.id

    stmt = select(Meeting).where(Meeting.organization_id == org_id)

    # Search filter (title or description)
    if search and search.strip():
        term = f"%{search.strip()}%"
        stmt = stmt.where(or_(Meeting.title.ilike(term), Meeting.description.ilike(term)))

    # Project filter
    if project_id and project_id.strip():
        stmt = stmt.where(Meeting.project_id == project_id.strip())

    # Category filter
    if category and category.strip():
        stmt = stmt.where(Meeting.category == category.strip())

    # Status filter
    if status and status.strip():
        stmt = stmt.where(Meeting.status == status.strip())

    # Start date filter
    if start_date:
        stmt = stmt.where(
            or_(
                Meeting.actual_start_time >= start_date,
                and_(Meeting.actual_start_time.is_(None), Meeting.created_at >= start_date)
            )
        )

    # End date filter
    if end_date:
        stmt = stmt.where(
            or_(
                Meeting.actual_start_time <= end_date,
                and_(Meeting.actual_start_time.is_(None), Meeting.created_at <= end_date)
            )
        )

    # Participant filter
    if participant and participant.strip():
        p_term = f"%{participant.strip()}%"
        subq = (
            select(MeetingParticipant.meeting_id)
            .where(
                or_(
                    MeetingParticipant.display_name.ilike(p_term),
                    MeetingParticipant.email.ilike(p_term)
                )
            )
        )
        stmt = stmt.where(Meeting.id.in_(subq))

    # Order by actual meeting time or creation time descending
    stmt = stmt.order_by(
        func.coalesce(Meeting.actual_start_time, Meeting.created_at).desc()
    )

    # Check if pagination is explicitly requested or parameters supplied
    is_paginated_request = paginate or (page is not None) or (limit is not None and limit < 100)

    if is_paginated_request:
        count_stmt = select(func.count(Meeting.id)).where(Meeting.organization_id == org_id)
        if search and search.strip():
            count_stmt = count_stmt.where(or_(Meeting.title.ilike(f"%{search.strip()}%"), Meeting.description.ilike(f"%{search.strip()}%")))
        if project_id and project_id.strip():
            count_stmt = count_stmt.where(Meeting.project_id == project_id.strip())
        if category and category.strip():
            count_stmt = count_stmt.where(Meeting.category == category.strip())
        if status and status.strip():
            count_stmt = count_stmt.where(Meeting.status == status.strip())
        if start_date:
            count_stmt = count_stmt.where(or_(Meeting.actual_start_time >= start_date, and_(Meeting.actual_start_time.is_(None), Meeting.created_at >= start_date)))
        if end_date:
            count_stmt = count_stmt.where(or_(Meeting.actual_start_time <= end_date, and_(Meeting.actual_start_time.is_(None), Meeting.created_at <= end_date)))
        if participant and participant.strip():
            count_stmt = count_stmt.where(Meeting.id.in_(select(MeetingParticipant.meeting_id).where(or_(MeetingParticipant.display_name.ilike(f"%{participant.strip()}%"), MeetingParticipant.email.ilike(f"%{participant.strip()}%")))))

        total = db.execute(count_stmt).scalar() or 0
        per_page = limit or 10

        if page is not None:
            current_page = page
            offset = (page - 1) * per_page
        else:
            current_page = (skip // per_page) + 1
            offset = skip

        pages = max(1, math.ceil(total / per_page)) if per_page > 0 else 1
        paginated_stmt = stmt.offset(offset).limit(per_page)
        meetings = db.execute(paginated_stmt).scalars().all()

        if paginate or page is not None:
            return PaginatedMeetingsResponse(
                items=meetings,
                total=total,
                page=current_page,
                pages=pages,
                limit=per_page
            )
        return meetings

    # Default unpaginated query for backwards compatibility
    meetings = db.execute(stmt).scalars().all()
    return meetings


@router.get(
    "/meetings/{meeting_id}",
    response_model=MeetingResponse,
    summary="Get Meeting Details",
    description="Get specific meeting details with media assets, jobs, and participants."
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


@router.delete(
    "/meetings/{meeting_id}",
    summary="Safely Delete Meeting",
    description="Permanently deletes a meeting, cascading related database records, cleaning vector and FTS5 memory chunks, and removing physical disk storage. Requires ORG_ADMIN."
)
def delete_meeting(
    meeting_id: str,
    tenant: TenantContext = Depends(require_org_admin),
    db: Session = Depends(get_db)
):
    org_id = tenant.organization.id
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == org_id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting not found."
        )

    title = meeting.title

    # 1. Clean vector and FTS5 chunks from memory storage
    try:
        ChunkStorageManager.delete_chunks_by_meeting(db, meeting_id=meeting_id, organization_id=org_id)
    except Exception as e:
        logger.warning(f"Failed to delete chunks for meeting {meeting_id}: {e}")

    # 2. Clean physical disk storage
    try:
        storage_dir = get_meeting_storage_dir(meeting_id)
        if storage_dir.exists():
            shutil.rmtree(storage_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to remove physical storage directory for meeting {meeting_id}: {e}")

    # 3. Log audit action
    audit = AuditLog(
        organization_id=org_id,
        user_id=tenant.user.id,
        action="MEETING_DELETED",
        entity_type="Meeting",
        entity_id=meeting_id,
        details=f"Meeting '{title}' and all associated transcripts, intelligence, and media files permanently deleted."
    )
    db.add(audit)

    # 4. Cascade delete meeting in DB
    db.delete(meeting)
    db.commit()

    return {"message": "Meeting and all associated resources safely deleted.", "meeting_id": meeting_id}


# --- Meeting Participants Endpoints ---

@router.get(
    "/meetings/{meeting_id}/participants",
    response_model=List[MeetingParticipantResponse],
    summary="List Meeting Participants",
    description="List all registered or guest participants in a meeting."
)
def list_meeting_participants(
    meeting_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> List[MeetingParticipantResponse]:
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found.")

    parts = db.execute(
        select(MeetingParticipant)
        .where(MeetingParticipant.meeting_id == meeting_id)
        .order_by(MeetingParticipant.created_at.asc())
    ).scalars().all()
    return parts


@router.post(
    "/meetings/{meeting_id}/participants",
    response_model=MeetingParticipantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add Meeting Participant",
    description="Add an attendee or participant to a meeting."
)
def add_meeting_participant(
    meeting_id: str,
    payload: MeetingParticipantCreate,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
) -> MeetingParticipantResponse:
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found.")

    part = MeetingParticipant(
        meeting_id=meeting_id,
        display_name=payload.display_name.strip(),
        email=payload.email.strip() if payload.email else None,
        role=payload.role.strip() if payload.role else None,
        user_id=payload.user_id,
    )
    db.add(part)
    db.commit()
    db.refresh(part)
    return part


@router.delete(
    "/meetings/{meeting_id}/participants/{participant_id}",
    summary="Remove Meeting Participant",
    description="Remove an attendee from a meeting."
)
def remove_meeting_participant(
    meeting_id: str,
    participant_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
):
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found.")

    part = db.execute(
        select(MeetingParticipant).where(
            MeetingParticipant.id == participant_id,
            MeetingParticipant.meeting_id == meeting_id,
        )
    ).scalar_one_or_none()
    if not part:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Participant not found.")

    db.delete(part)
    db.commit()
    return {"message": "Participant removed successfully", "id": participant_id}


# --- Synchronized Audio Streaming Endpoint ---

@router.get(
    "/meetings/{meeting_id}/audio",
    summary="Stream Meeting Audio",
    description="Streams meeting audio with HTTP Range support for synchronized playback. Returns 410 Gone if audio was auto-deleted per privacy policy."
)
def stream_meeting_audio(
    meeting_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
):
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found.")

    # Find the best audio asset (prefer normalized WAV, then raw asset)
    asset = db.execute(
        select(MediaAsset)
        .where(MediaAsset.meeting_id == meeting_id)
        .order_by(MediaAsset.is_raw_deleted.asc(), MediaAsset.created_at.desc())
    ).scalars().first()

    if not asset or asset.is_raw_deleted:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Audio recording has been safely deleted per the organization's privacy and retention policy."
        )

    file_path = resolve_safe_storage_path(asset.storage_path)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Audio recording is not available on disk (retention policy expired)."
        )

    return FileResponse(
        path=file_path,
        media_type=asset.mime_type or "audio/wav",
        filename=asset.file_name
    )


@router.get(
    "/meetings/{meeting_id}/export/pdf",
    summary="Export Meeting Intelligence Report as PDF",
    description="Generates a downloadable, publication-grade executive report PDF containing meeting metadata, participants, executive summary, topics, decisions, action items, risks, and transcript evidence.",
)
def export_meeting_pdf_report(
    meeting_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db)
):
    """Generate and download a comprehensive meeting executive report PDF."""
    meeting = db.execute(
        select(Meeting).where(
            Meeting.id == meeting_id,
            Meeting.organization_id == tenant.organization.id,
        )
    ).scalar_one_or_none()
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting not found in this organization."
        )

    # Project lookup
    project_name = None
    if meeting.project_id:
        project = db.execute(
            select(Project).where(
                Project.id == meeting.project_id,
                Project.organization_id == tenant.organization.id,
            )
        ).scalar_one_or_none()
        if project:
            project_name = project.name

    # Participants
    participants_query = db.execute(
        select(MeetingParticipant).where(MeetingParticipant.meeting_id == meeting.id)
    ).scalars().all()
    participants_data = [
        {
            "display_name": p.display_name,
            "email": p.email,
            "role": p.role,
            "is_speaker": getattr(p, "is_speaker", False),
        }
        for p in participants_query
    ]

    # Intelligence structures
    summary = db.query(MeetingSummary).filter(MeetingSummary.meeting_id == meeting.id).first()
    topics = db.query(Topic).filter(Topic.meeting_id == meeting.id).order_by(Topic.start_time.asc()).all()
    decisions = db.query(Decision).filter(Decision.meeting_id == meeting.id).order_by(Decision.timestamp_seconds.asc()).all()
    action_items = db.query(ActionItem).filter(ActionItem.meeting_id == meeting.id).order_by(ActionItem.timestamp_seconds.asc()).all()
    questions = db.query(UnresolvedQuestion).filter(UnresolvedQuestion.meeting_id == meeting.id).order_by(UnresolvedQuestion.created_at.asc()).all()
    risks = db.query(Risk).filter(Risk.meeting_id == meeting.id).order_by(Risk.created_at.asc()).all()

    intelligence_data = {
        "summary": {
            "summary_text": (summary.executive_summary or summary.detailed_summary) if summary else None,
            "executive_summary": summary.executive_summary if summary else None,
            "detailed_summary": summary.detailed_summary if summary else None,
            "key_takeaways": summary.key_takeaways if summary else None,
        } if summary else None,
        "topics": [
            {"title": t.title, "description": t.description, "start_time": t.start_time, "end_time": t.end_time}
            for t in topics
        ],
        "decisions": [
            {
                "decision_text": getattr(d, "description", None) or getattr(d, "decision_text", ""),
                "status": d.status,
                "rationale": getattr(d, "rationale_or_context", None) or getattr(d, "rationale", ""),
                "timestamp_seconds": d.timestamp_seconds,
                "evidence_segment_id": d.evidence_segment_id,
            }
            for d in decisions
        ],
        "action_items": [
            {
                "description": a.description,
                "owner_name": a.owner_name,
                "status": a.status,
                "deadline": a.deadline.isoformat() if a.deadline else None,
                "timestamp_seconds": a.timestamp_seconds,
            }
            for a in action_items
        ],
        "questions": [
            {"question_text": q.question_text, "raised_by": q.raised_by, "status": q.status}
            for q in questions
        ],
        "risks": [
            {"description": r.description, "severity": r.severity}
            for r in risks
        ],
    }

    # Verified transcript highlights (prefer privacy-safe redacted transcript)
    safe_transcript = (
        db.query(Transcript)
        .filter(
            Transcript.meeting_id == meeting.id,
            Transcript.transcript_type.in_([
                TranscriptType.CLEANED_REDACTED.value,
                TranscriptType.REDACTED.value,
                TranscriptType.ORIGINAL.value,
            ])
        )
        .order_by(Transcript.created_at.desc())
        .first()
    )
    evidence_snippets = []
    if safe_transcript:
        segments = (
            db.query(TranscriptSegment)
            .filter(TranscriptSegment.transcript_id == safe_transcript.id)
            .order_by(TranscriptSegment.start_time.asc())
            .limit(10)
            .all()
        )
        evidence_snippets = [
            {
                "start_time": s.start_time,
                "end_time": s.end_time,
                "speaker": getattr(s, "speaker_label", None) or "Participant",
                "text": s.text,
            }
            for s in segments
        ]

    # Meeting info payload
    meeting_data = {
        "title": meeting.title,
        "description": meeting.description,
        "actual_start_time": meeting.actual_start_time.isoformat() if meeting.actual_start_time else None,
        "scheduled_start_time": meeting.scheduled_start_time.isoformat() if meeting.scheduled_start_time else None,
        "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
        "project_name": project_name,
        "category": meeting.category,
        "status": meeting.status,
        "input_source_type": meeting.input_source_type,
    }

    # Generate PDF
    pdf_bytes = generate_meeting_pdf_report(
        meeting_data=meeting_data,
        intelligence_data=intelligence_data,
        participants_data=participants_data,
        evidence_snippets=evidence_snippets,
    )

    clean_title = "".join(c for c in meeting.title if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
    filename = f"MEMORA_Report_{clean_title[:40] or 'Meeting'}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )


# --- Processing Jobs & Safe Retry ---

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


@router.post(
    "/jobs/{job_id}/retry",
    response_model=ProcessingJobResponse,
    summary="Retry Failed Processing Job",
    description="Safely retries an eligible failed processing job if media is available and retry limits not exceeded."
)
def retry_processing_job(
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

    if job.status != JobStatus.FAILED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job cannot be retried because its status is {job.status}. Only FAILED jobs are eligible for retry."
        )

    max_retries = job.max_attempts or 3
    if job.attempt_count >= max_retries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job has reached the maximum allowed retry limit ({job.attempt_count}/{max_retries})."
        )

    # For transcription or ingestion jobs, verify media is still available
    if job.job_type in (JobType.INGESTION.value, JobType.TRANSCRIPTION.value):
        asset = db.execute(
            select(MediaAsset).where(
                MediaAsset.meeting_id == job.meeting_id,
                MediaAsset.is_raw_deleted == False
            )
        ).scalars().first()

        if not asset:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot retry transcription: raw media file was deleted per privacy policy."
            )

        disk_path = resolve_safe_storage_path(asset.storage_path)
        if not disk_path or not disk_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot retry transcription: raw media file is missing from storage."
            )

    # Reset job to PENDING so JobWorker picks it up
    now = datetime.now(timezone.utc)
    job.status = JobStatus.PENDING.value
    job.progress = 0.0
    job.error_message = None
    job.completed_at = None
    job.updated_at = now

    reconcile_meeting_status(db, job.meeting_id)
    db.commit()
    db.refresh(job)

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
