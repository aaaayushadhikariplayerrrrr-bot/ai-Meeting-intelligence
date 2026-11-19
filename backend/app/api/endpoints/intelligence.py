"""Structured Meeting Intelligence API Endpoints.

Provides endpoints to:
1. Trigger asynchronous meeting intelligence extraction jobs (JobType.EXTRACTION).
2. Retrieve comprehensive structured meeting intelligence (summary, topics, decisions, actions, risks, questions).
3. Update action-item status with audit logging and strict tenant isolation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.api.deps import get_current_tenant_membership, TenantContext
from app.core.config import settings
from app.models.meeting import Meeting, ProcessingJob, JobType, JobStatus
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
    ActionItemStatus,
)
from app.models.transcription import Transcript, TranscriptType
from app.models.memory import AuditLog
from app.schemas.intelligence import (
    ExtractIntelligenceRequest,
    ExtractIntelligenceJobResponse,
    MeetingIntelligenceResponse,
    MeetingSummaryResponse,
    TopicResponse,
    DecisionResponse,
    ActionItemResponse,
    RiskResponse,
    UnresolvedQuestionResponse,
    ActionItemStatusUpdate,
)
from app.services.job_manager import create_processing_job

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Intelligence"])


@router.post(
    "/meetings/{meeting_id}/intelligence/extract",
    response_model=ExtractIntelligenceJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger asynchronous structured meeting intelligence extraction",
)
def extract_meeting_intelligence_job(
    meeting_id: str,
    payload: Optional[ExtractIntelligenceRequest] = None,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
) -> ExtractIntelligenceJobResponse:
    """Trigger an asynchronous background job to extract structured intelligence from a meeting.

    STRICT CONSTRAINTS:
    - Requires active organization membership.
    - Strictly scoped to the authenticated tenant organization.
    - Validates presence of at least one privacy-safe transcript (REDACTED or CLEANED_REDACTED).
    - Prevents duplicate active extraction jobs for the same meeting.
    - Offloads inference exclusively to the database-backed JobWorker (never executes LLM in request).
    """
    if payload is None:
        payload = ExtractIntelligenceRequest()

    # 1. Tenant-scoped meeting lookup
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found in this organization."
        )

    # 2. Validate provider override against configuration
    configured_provider = getattr(settings, "INTELLIGENCE_PROVIDER", "deterministic").strip().lower()
    selected_provider = (payload.provider_override or configured_provider).strip().lower()

    if selected_provider == "disabled" or configured_provider == "disabled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Structured meeting intelligence extraction is disabled in configuration."
        )

    if selected_provider not in {"gemini", "ollama", "deterministic"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider_override '{payload.provider_override}'. Allowed values: 'gemini', 'ollama', 'deterministic'."
        )

    if selected_provider == "gemini":
        if not getattr(settings, "GEMINI_ENABLED", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gemini intelligence provider is disabled. Please set GEMINI_ENABLED=true in configuration."
            )
        if not getattr(settings, "GEMINI_API_KEY", "") or not getattr(settings, "GEMINI_API_KEY", "").strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gemini API key is not configured. Please configure GEMINI_API_KEY in the backend environment."
            )

    # 3. Privacy-safe transcript validation
    if payload.source_transcript_id:
        target_transcript = (
            db.query(Transcript)
            .filter(Transcript.id == payload.source_transcript_id, Transcript.meeting_id == meeting.id)
            .first()
        )
        if not target_transcript:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Specified transcript '{payload.source_transcript_id}' not found for meeting '{meeting.id}'."
            )
        if target_transcript.transcript_type not in (
            TranscriptType.REDACTED.value,
            TranscriptType.CLEANED_REDACTED.value,
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Transcript '{target_transcript.id}' is of type '{target_transcript.transcript_type}'. "
                    "Intelligence extraction strictly requires privacy-safe transcripts (REDACTED or CLEANED_REDACTED)."
                )
            )
    else:
        # Check that at least one privacy-safe transcript exists for this meeting
        safe_transcript_exists = (
            db.query(Transcript.id)
            .filter(
                Transcript.meeting_id == meeting.id,
                Transcript.transcript_type.in_([
                    TranscriptType.CLEANED_REDACTED.value,
                    TranscriptType.REDACTED.value,
                ])
            )
            .first()
        )
        if not safe_transcript_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"No privacy-safe (redacted) transcript available for meeting '{meeting.id}'. "
                    "Please generate a REDACTED or CLEANED_REDACTED transcript version before running intelligence extraction."
                )
            )

    # 4. Duplicate active extraction job prevention
    active_job = (
        db.query(ProcessingJob)
        .filter(
            ProcessingJob.meeting_id == meeting.id,
            ProcessingJob.job_type == JobType.EXTRACTION.value,
            ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value])
        )
        .first()
    )
    if active_job:
        return ExtractIntelligenceJobResponse(
            job_id=active_job.id,
            meeting_id=meeting.id,
            job_type=active_job.job_type,
            status=active_job.status,
            message=f"An extraction job is already active for meeting '{meeting.id}'.",
        )

    # 5. Persist validated execution parameters in job payload
    job_payload = {
        "meeting_id": meeting.id,
        "source_transcript_id": payload.source_transcript_id,
        "output_language": payload.output_language or "en",
        "provider_override": payload.provider_override,
        "organization_id": tenant.organization.id,
    }

    job = create_processing_job(
        db=db,
        meeting_id=meeting.id,
        job_type=JobType.EXTRACTION.value,
        initial_status=JobStatus.PENDING.value,
        payload=job_payload,
    )

    logger.info(
        f"Enqueued EXTRACTION job '{job.id}' for meeting '{meeting.id}' "
        f"by user '{tenant.user.id}' (org: '{tenant.organization.id}')."
    )

    return ExtractIntelligenceJobResponse(
        job_id=job.id,
        meeting_id=meeting.id,
        job_type=job.job_type,
        status=job.status,
        message="Structured meeting intelligence extraction job queued successfully.",
    )


@router.get(
    "/meetings/{meeting_id}/intelligence",
    response_model=MeetingIntelligenceResponse,
    summary="Retrieve structured intelligence for a meeting",
)
def get_meeting_intelligence(
    meeting_id: str,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
) -> MeetingIntelligenceResponse:
    """Retrieve synthesized executive summary, topics, decisions, action items, risks, and questions.

    STRICT CONSTRAINTS:
    - Scoped strictly to the authenticated tenant organization.
    - Returns consistent empty structures if intelligence has not yet been extracted.
    - Never exposes raw unredacted transcript text or model prompts.
    - Avoids N+1 queries.
    """
    meeting = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Meeting '{meeting_id}' not found in this organization."
        )

    # Query latest summary and intelligence categories in batch
    summary = (
        db.query(MeetingSummary)
        .filter(MeetingSummary.meeting_id == meeting.id)
        .first()
    )
    topics = (
        db.query(Topic)
        .filter(Topic.meeting_id == meeting.id)
        .order_by(Topic.start_time.asc())
        .all()
    )
    decisions = (
        db.query(Decision)
        .filter(Decision.meeting_id == meeting.id)
        .order_by(Decision.timestamp_seconds.asc())
        .all()
    )
    action_items = (
        db.query(ActionItem)
        .filter(ActionItem.meeting_id == meeting.id)
        .order_by(ActionItem.timestamp_seconds.asc())
        .all()
    )
    risks = (
        db.query(Risk)
        .filter(Risk.meeting_id == meeting.id)
        .order_by(Risk.created_at.asc())
        .all()
    )
    questions = (
        db.query(UnresolvedQuestion)
        .filter(UnresolvedQuestion.meeting_id == meeting.id)
        .order_by(UnresolvedQuestion.created_at.asc())
        .all()
    )

    # Return empty response if no intelligence has been extracted
    if not summary and not (topics or decisions or action_items or risks or questions):
        return MeetingIntelligenceResponse(
            meeting_id=meeting.id,
            summary=None,
            topics=[],
            decisions=[],
            action_items=[],
            risks=[],
            unresolved_questions=[],
            has_intelligence=False,
        )

    summary_resp = MeetingSummaryResponse.model_validate(summary) if summary else None
    topics_resp = [TopicResponse.model_validate(t) for t in topics]
    decisions_resp = [DecisionResponse.model_validate(d) for d in decisions]
    actions_resp = [ActionItemResponse.model_validate(a) for a in action_items]
    risks_resp = [RiskResponse.model_validate(r) for r in risks]
    questions_resp = [UnresolvedQuestionResponse.model_validate(q) for q in questions]

    return MeetingIntelligenceResponse(
        meeting_id=meeting.id,
        summary=summary_resp,
        topics=topics_resp,
        decisions=decisions_resp,
        action_items=actions_resp,
        risks=risks_resp,
        unresolved_questions=questions_resp,
        provider_used=summary.provider_used if summary else None,
        is_heuristic=summary.is_heuristic if summary else None,
        fallback_warning=summary.fallback_warning if summary else None,
    )


@router.patch(
    "/action-items/{item_id}/status",
    response_model=ActionItemResponse,
    summary="Update the lifecycle status of an action item",
)
def update_action_item_status(
    item_id: str,
    payload: ActionItemStatusUpdate,
    request: Request,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant_membership),
) -> ActionItemResponse:
    """Update only the status of an action item within the authenticated tenant context.

    STRICT CONSTRAINTS:
    - Scoped strictly to meetings belonging to the authenticated tenant organization.
    - Preserves description, owner, deadline, and evidence segment associations.
    - Creates an immutable audit log entry for compliance.
    """
    action_item = (
        db.query(ActionItem)
        .join(Meeting, ActionItem.meeting_id == Meeting.id)
        .filter(ActionItem.id == item_id, Meeting.organization_id == tenant.organization.id)
        .first()
    )
    if not action_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Action item '{item_id}' not found in this organization."
        )

    old_status = action_item.status
    new_status = payload.status.value if isinstance(payload.status, ActionItemStatus) else str(payload.status)

    action_item.status = new_status
    action_item.updated_at = datetime.now(timezone.utc)

    # Immutable audit trail
    client_ip = request.client.host if request and request.client else None
    audit_details = {
        "meeting_id": action_item.meeting_id,
        "action_item_id": action_item.id,
        "old_status": old_status,
        "new_status": new_status,
    }
    audit_entry = AuditLog(
        organization_id=tenant.organization.id,
        user_id=tenant.user.id,
        action="UPDATE_ACTION_ITEM_STATUS",
        entity_type="ActionItem",
        entity_id=action_item.id,
        details=json.dumps(audit_details),
        ip_address=client_ip,
    )
    db.add(audit_entry)
    db.commit()
    db.refresh(action_item)

    logger.info(
        f"Action item '{item_id}' status updated: '{old_status}' -> '{new_status}' "
        f"by user '{tenant.user.id}' (org: '{tenant.organization.id}')."
    )

    return ActionItemResponse.model_validate(action_item)
