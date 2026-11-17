"""Transcription API Endpoints.

Provides endpoints to trigger multilingual speech recognition jobs,
retrieve transcript versions and timestamped segments, and query ASR hardware diagnostics.
"""

import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.database.session import get_db
from app.api.deps import (
    get_current_tenant_membership,
    require_org_admin,
    require_super_admin,
    TenantContext,
)
from app.models.identity import Organization, User, OrgRole
from app.models.meeting import Meeting, ProcessingJob, JobType, JobStatus
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.memory import AuditLog
from app.services.job_manager import create_processing_job
from app.services.transcription.job_handler import process_transcription_job_sync
from app.services.transcription.gpu_manager import (
    check_gpu_hardware,
    check_ctranslate2_support,
    get_cpu_resource_info,
)
from app.services.transcription.manager import get_asr_engine
from app.services.transcript_processing.processor import derive_processed_transcript
from app.core.config import settings

router = APIRouter(prefix="/transcription", tags=["transcription"])


# --- Schemas ---
class StartTranscriptionRequest(BaseModel):
    language: Optional[str] = Field(
        None,
        description="Optional ISO language code (e.g. 'en', 'hi', 'gu'). Omit for auto-detection."
    )
    transcription_mode: Optional[str] = Field(
        "standard",
        description="Application transcription mode: 'standard' or 'hinglish' (Hindi-English code-switching)."
    )


class CleanTranscriptRequest(BaseModel):
    source_transcript_id: Optional[str] = Field(
        None,
        description="Optional specific source transcript ID. Defaults to latest ORIGINAL transcript."
    )


class RedactTranscriptRequest(BaseModel):
    source_transcript_id: Optional[str] = Field(
        None,
        description="Optional specific source transcript ID. Defaults to latest ORIGINAL transcript."
    )


class ProcessTranscriptRequest(BaseModel):
    source_transcript_id: Optional[str] = Field(
        None,
        description="Optional specific source transcript ID. Defaults to latest ORIGINAL transcript."
    )
    clean: bool = Field(True, description="Enable transcript readability cleaning and filler reduction")
    redact: bool = Field(True, description="Enable privacy detection and PII redaction")


class SegmentResponse(BaseModel):
    id: str
    sequence_order: int
    start_time: float
    end_time: float
    text: str
    language_code: str
    confidence: Optional[float] = Field(
        None,
        description="Uncalibrated confidence score derived from exp(avg_logprob), bounded [0.0, 1.0]. Not a calibrated probability."
    )
    speaker_label: Optional[str] = None
    is_redacted: bool = False
    source_segment_id: Optional[str] = None
    redaction_metadata: Optional[List[str]] = None


class TranscriptResponse(BaseModel):
    id: str
    meeting_id: str
    transcript_type: str
    version: int
    language_detected: Optional[str]
    is_primary: bool
    parent_transcript_id: Optional[str] = None
    created_at: str
    segments: List[SegmentResponse] = []


class JobStatusResponse(BaseModel):
    job_id: str
    meeting_id: str
    job_type: str
    status: str
    progress: float
    error_message: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 2


class HardwareStatusResponse(BaseModel):
    gpu_info: Dict[str, Any]
    ctranslate2_cuda_support: bool
    ctranslate2_message: str
    ctranslate2_supported_compute_types: List[str]
    active_engine: Dict[str, Any]
    cpu_info: Optional[Dict[str, Any]] = None


# --- Endpoints ---
@router.post("/start/{meeting_id}", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
def start_transcription(
    meeting_id: str,
    payload: Optional[StartTranscriptionRequest] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Start an asynchronous multilingual transcription job for a meeting's media."""
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    # Check that non-deleted media assets exist
    valid_assets = [a for a in meeting.media_assets if not a.is_raw_deleted]
    if not valid_assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Meeting has no valid media assets available for transcription."
        )

    # Reject media exceeding the configured limit before creating a processing job or starting transcription
    for asset in valid_assets:
        if asset.duration_seconds is not None and asset.duration_seconds > settings.MAX_MEDIA_DURATION_SECONDS:
            max_hrs = settings.MAX_MEDIA_DURATION_SECONDS / 3600.0
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Media asset duration ({asset.duration_seconds:.1f}s) exceeds maximum allowed limit of {settings.MAX_MEDIA_DURATION_SECONDS:.1f}s ({max_hrs:.1f} hours)."
            )

    # Check if a transcription job is already active
    active_job = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting.id,
            ProcessingJob.job_type == JobType.TRANSCRIPTION.value,
            ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value])
        )
        .first()
    )
    if active_job:
        return JobStatusResponse(
            job_id=active_job.id,
            meeting_id=meeting.id,
            job_type=active_job.job_type,
            status=active_job.status,
            progress=active_job.progress,
            error_message=active_job.error_message,
            attempt_count=active_job.attempt_count,
            max_attempts=active_job.max_attempts,
        )

    # Persist execution parameters in job payload
    lang = None
    mode = "standard"
    if payload:
        if payload.language in ("hi-en", "hinglish"):
            mode = "hinglish"
            lang = None
        else:
            lang = payload.language
            mode = payload.transcription_mode or "standard"

    job_payload = {
        "language_override": lang,
        "transcription_mode": mode,
    }

    # Create new persistent ProcessingJob for background worker execution
    job = create_processing_job(
        db=db,
        meeting_id=meeting.id,
        job_type=JobType.TRANSCRIPTION.value,
        initial_status=JobStatus.PENDING.value,
        payload=job_payload,
    )

    return JobStatusResponse(
        job_id=job.id,
        meeting_id=meeting.id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
    )


def _serialize_transcript(t: Transcript) -> TranscriptResponse:
    """Format ORM Transcript model to TranscriptResponse schema."""
    segs = [
        SegmentResponse(
            id=s.id,
            sequence_order=s.sequence_order,
            start_time=s.start_time,
            end_time=s.end_time,
            text=s.text,
            language_code=s.language_code,
            confidence=s.confidence,
            speaker_label=s.speaker.speaker_label if s.speaker else None,
            is_redacted=bool(s.is_redacted),
            source_segment_id=s.source_segment_id,
            redaction_metadata=json.loads(s.redaction_metadata) if s.redaction_metadata else None,
        )
        for s in t.segments
    ]
    return TranscriptResponse(
        id=t.id,
        meeting_id=t.meeting_id,
        transcript_type=t.transcript_type,
        version=t.version,
        language_detected=t.language_detected,
        is_primary=t.is_primary,
        parent_transcript_id=t.parent_transcript_id,
        created_at=t.created_at.isoformat() if t.created_at else "",
        segments=segs,
    )


@router.get("/{meeting_id}/default", response_model=TranscriptResponse)
def get_default_transcript_for_meeting(
    meeting_id: str,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Retrieve the single default, privacy-safe transcript for a meeting.

    STRICT PRIVACY RULES:
    1. Prefer CLEANED_REDACTED (latest version).
    2. Otherwise use REDACTED (latest version).
    3. If neither exists, return HTTP 404 indicating no privacy-safe transcript is available.
    4. NEVER return CLEANED or ORIGINAL from this endpoint.
    """
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    # Strictly select only privacy-safe (redacted) transcript types
    privacy_safe_transcripts = (
        db.query(Transcript)
        .filter(
            Transcript.meeting_id == meeting.id,
            Transcript.transcript_type.in_([
                TranscriptType.CLEANED_REDACTED.value,
                TranscriptType.REDACTED.value,
            ])
        )
        .all()
    )
    if not privacy_safe_transcripts:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No privacy-safe (redacted) transcript available for meeting {meeting_id}. "
                "Only unredacted transcripts exist. Generate a REDACTED or CLEANED_REDACTED version first."
            )
        )

    # Priority sorting: CLEANED_REDACTED (0) > REDACTED (1), picking the latest version
    priority_map = {
        TranscriptType.CLEANED_REDACTED.value: 0,
        TranscriptType.REDACTED.value: 1,
    }
    selected = sorted(
        privacy_safe_transcripts,
        key=lambda t: (priority_map.get(t.transcript_type, 99), -t.version)
    )[0]

    return _serialize_transcript(selected)


@router.get("/{meeting_id}", response_model=List[TranscriptResponse])
def get_transcripts_for_meeting(
    meeting_id: str,
    transcript_type: Optional[str] = Query(None, description="Transcript version type (ORIGINAL, CLEANED, REDACTED, CLEANED_REDACTED, or omit for all)"),
    version: Optional[int] = Query(None, description="Specific version number"),
    request: Request = None,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Retrieve saved transcripts and timestamped segments for a meeting.
    
    Role-Based Access Control:
    - ORG_ADMIN: May retrieve ORIGINAL, CLEANED, REDACTED, and CLEANED_REDACTED.
    - MEMBER: May ONLY retrieve privacy-safe transcripts (REDACTED, CLEANED_REDACTED).
      Explicit requests for ORIGINAL or CLEANED return HTTP 403 Forbidden.
    """
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    is_admin = (tenant.membership.role == OrgRole.ORG_ADMIN.value)
    norm_type = transcript_type.upper() if transcript_type else None

    if not is_admin:
        if norm_type in [TranscriptType.ORIGINAL.value, TranscriptType.CLEANED.value]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to unredacted original or cleaned transcripts requires organization administrator privileges.",
            )
        # Filter to privacy-safe transcripts only
        query = db.query(Transcript).filter(
            Transcript.meeting_id == meeting.id,
            Transcript.transcript_type.in_([
                TranscriptType.REDACTED.value,
                TranscriptType.CLEANED_REDACTED.value,
            ])
        )
        if norm_type and norm_type != "ALL":
            query = query.filter(Transcript.transcript_type == norm_type)
    else:
        query = db.query(Transcript).filter(Transcript.meeting_id == meeting.id)
        if norm_type and norm_type != "ALL":
            query = query.filter(Transcript.transcript_type == norm_type)

    if version is not None:
        query = query.filter(Transcript.version == version)

    transcripts = query.order_by(Transcript.created_at.desc(), Transcript.version.desc()).all()
    serialized = [_serialize_transcript(t) for t in transcripts]

    # If an authorized ORG_ADMIN retrieved any ORIGINAL or CLEANED transcripts, record an audit event
    if is_admin:
        unredacted_retrieved = [
            t for t in transcripts
            if t.transcript_type in (TranscriptType.ORIGINAL.value, TranscriptType.CLEANED.value)
        ]
        if unredacted_retrieved:
            client_ip = request.client.host if request and request.client else None
            audit_details = {
                "meeting_id": meeting.id,
                "retrieved_transcript_ids": [t.id for t in unredacted_retrieved],
                "retrieved_types": [t.transcript_type for t in unredacted_retrieved],
                "requested_transcript_type": norm_type,
                "requested_version": version,
                "unredacted_count": len(unredacted_retrieved),
            }
            audit_entry = AuditLog(
                organization_id=tenant.organization.id,
                user_id=tenant.user.id,
                action="VIEW_UNREDACTED_TRANSCRIPT",
                entity_type="Meeting",
                entity_id=meeting.id,
                details=json.dumps(audit_details),
                ip_address=client_ip,
            )
            db.add(audit_entry)
            db.commit()

    return serialized


@router.post("/{meeting_id}/clean", response_model=TranscriptResponse, status_code=status.HTTP_201_CREATED)
def clean_meeting_transcript(
    meeting_id: str,
    payload: Optional[CleanTranscriptRequest] = None,
    request: Request = None,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(require_org_admin),
):
    """Generate a derived CLEANED transcript version with filler and stutter reduction.

    Role-Based Access Control:
    Requires ORG_ADMIN role because CLEANED transcripts contain unredacted PII and secrets.
    """
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    ip = request.client.host if request and request.client else None
    source_id = payload.source_transcript_id if payload else None
    derived = derive_processed_transcript(
        meeting_id=meeting_id,
        organization_id=tenant.organization.id,
        target_type=TranscriptType.CLEANED,
        db=db,
        source_transcript_id=source_id,
        user_id=tenant.user.id,
        ip_address=ip,
    )
    return _serialize_transcript(derived)


@router.post("/{meeting_id}/redact", response_model=TranscriptResponse, status_code=status.HTTP_201_CREATED)
def redact_meeting_transcript(
    meeting_id: str,
    payload: Optional[RedactTranscriptRequest] = None,
    request: Request = None,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Generate a derived REDACTED transcript version with PII and secret redaction."""
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    ip = request.client.host if request and request.client else None
    source_id = payload.source_transcript_id if payload else None
    derived = derive_processed_transcript(
        meeting_id=meeting_id,
        organization_id=tenant.organization.id,
        target_type=TranscriptType.REDACTED,
        db=db,
        source_transcript_id=source_id,
        user_id=tenant.user.id,
        ip_address=ip,
    )
    return _serialize_transcript(derived)


@router.post("/{meeting_id}/process", response_model=TranscriptResponse, status_code=status.HTTP_201_CREATED)
def process_meeting_transcript(
    meeting_id: str,
    payload: Optional[ProcessTranscriptRequest] = None,
    request: Request = None,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Generate a derived processed transcript version (CLEANED, REDACTED, or CLEANED_REDACTED).

    Role-Based Access Control:
    - ORG_ADMIN: May generate any derived transcript version (CLEANED, REDACTED, CLEANED_REDACTED).
    - MEMBER: May ONLY generate privacy-safe derived versions (REDACTED, CLEANED_REDACTED).
      Requests resulting in unredacted CLEANED transcripts (redact=False) return HTTP 403 Forbidden.
    """
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting {meeting_id} not found."
        )

    clean = payload.clean if payload else True
    redact = payload.redact if payload else True
    source_id = payload.source_transcript_id if payload else None

    if clean and redact:
        target_type = TranscriptType.CLEANED_REDACTED
    elif clean:
        target_type = TranscriptType.CLEANED
    elif redact:
        target_type = TranscriptType.REDACTED
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of 'clean' or 'redact' must be enabled."
        )

    # Role check: CLEANED transcript produces unredacted content containing PII/secrets.
    # Generating unredacted transcripts requires ORG_ADMIN.
    if target_type == TranscriptType.CLEANED and tenant.membership.role != OrgRole.ORG_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization administrator privileges required to generate unredacted transcripts.",
        )

    ip = request.client.host if request and request.client else None
    derived = derive_processed_transcript(
        meeting_id=meeting_id,
        organization_id=tenant.organization.id,
        target_type=target_type,
        db=db,
        source_transcript_id=source_id,
        user_id=tenant.user.id,
        ip_address=ip,
    )
    return _serialize_transcript(derived)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_transcription_job_status(
    job_id: str,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
):
    """Get the current progress and status of a transcription job."""
    # Ensure organization isolation by joining Meeting
    job = (
        db.query(ProcessingJob)
        .join(Meeting, ProcessingJob.meeting_id == Meeting.id)
        .filter(ProcessingJob.id == job_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )

    return JobStatusResponse(
        job_id=job.id,
        meeting_id=job.meeting_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
    )


@router.get("/hardware/diagnostics", response_model=HardwareStatusResponse)
def get_hardware_diagnostics(
    current_user: User = Depends(require_super_admin),
):
    """Inspect GPU, CUDA, CTranslate2, and active ASR engine readiness.
    
    Protected platform-level diagnostics: requires global Super Admin privilege.
    """
    gpu_info = check_gpu_hardware()
    ct2_supported, ct2_msg, ct2_types = check_ctranslate2_support()
    engine = get_asr_engine()

    return HardwareStatusResponse(
        gpu_info=gpu_info,
        ctranslate2_cuda_support=ct2_supported,
        ctranslate2_message=ct2_msg,
        ctranslate2_supported_compute_types=ct2_types,
        active_engine=engine.get_engine_info(),
        cpu_info=get_cpu_resource_info(),
    )
