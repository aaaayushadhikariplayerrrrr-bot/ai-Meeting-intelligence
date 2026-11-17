"""Secure REST API Endpoints for Cross-Meeting Intelligence & Conflict Detection.

Provides tenant-scoped endpoints for:
1. Decisions:
   - GET /decisions: List decisions across organization with filters (status, meeting, dates, search).
   - GET /decisions/{decision_id}: Get decision details and complete historical lineage.
   - POST /decisions/{decision_id}/supersede: Explicitly supersede an existing decision with a new decision.
   - POST /decisions/{decision_id}/link-superseded: Explicitly link an existing decision as superseded by this decision.
   - POST /decisions/{decision_id}/revoke: Revoke an active decision with mandatory audited reason.

2. Commitments (Action Items & Reviews):
   - GET /commitments: List commitments across meetings with filters (status, owner, overdue, deadlines, meeting).
   - GET /commitments/{action_item_id}: Get commitment details and chronological review history.
   - POST /commitments/{action_item_id}/reviews: Record a commitment review event across meetings.
   - GET /commitments/{action_item_id}/reviews: List all review events for a commitment.

3. Unresolved Questions:
   - GET /questions: List questions across organization with filters (status, meeting, dates, search).
   - GET /questions/{question_id}: Get question details and resolution status.
   - POST /questions/{question_id}/resolve: Resolve an open question with resolving meeting, decision, and notes.
   - POST /questions/{question_id}/reopen: Reopen a question back to OPEN status.

4. Conflicts:
   - GET /conflicts: Detect potential contradictions across active decisions with heuristic attribution.

Security & Tenant Isolation Guarantees:
- Organization identity derived strictly from authenticated user context (TenantContext).
- Zero trust of client-supplied organization IDs.
- Validates every referenced meeting, decision, commitment, question, and evidence segment against the authenticated tenant.
- Safe HTTP 404 responses for non-existent and foreign-tenant resources to prevent enumeration.
- Preserves immutable audit logging for all mutations.
- Enforces rate limiting per client IP.
- Never exposes unredacted or original transcript text.
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import TenantContext, get_current_tenant_membership
from app.core.rate_limit import cross_meeting_rate_limiter
from app.database.session import get_db
from app.models.intelligence import ActionItem, ActionItemStatus, Decision, DecisionStatus, QuestionStatus, UnresolvedQuestion
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment
from app.schemas.intelligence import (
    ActionItemReviewResponse,
    CommitmentReviewUpdate,
    CrossMeetingCommitmentResponse,
    CrossMeetingDecisionResponse,
    CrossMeetingQuestionResponse,
    DecisionCreate,
    DecisionLinkSupersededRequest,
    DecisionRevokeRequest,
    DecisionSupersedeCreate,
    QuestionResolutionUpdate,
)
from app.schemas.memory import DecisionConflictItem
from app.services.intelligence.cross_meeting import (
    CommitmentTrackingService,
    DecisionLifecycleService,
    UnresolvedQuestionService,
)
from app.services.intelligence.lineage import (
    CircularLineageError,
    CrossTenantLineageError,
    LineageValidationError,
    SelfSupersedingError,
)
from app.services.memory.conflict_detector import ConflictDetector

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_client_ip(request: Request) -> Optional[str]:
    """Helper to extract direct client peer IP address."""
    if request and request.client and request.client.host:
        return request.client.host
    return None


# ===========================================================================
# 1. Decisions Endpoints
# ===========================================================================

@router.get(
    "/decisions",
    response_model=List[CrossMeetingDecisionResponse],
    summary="List decisions across organization",
    description="List decisions with optional status, meeting, date range, and keyword search filters.",
)
def list_decisions(
    status_filter: Optional[str] = Query(None, alias="status", description="ACTIVE, SUPERSEDED, or REVOKED"),
    meeting_id: Optional[str] = Query(None, description="Filter by origin meeting UUID"),
    start_date: Optional[datetime] = Query(None, description="Lower bound on creation date"),
    end_date: Optional[datetime] = Query(None, description="Upper bound on creation date"),
    search: Optional[str] = Query(None, description="Text search in description, rationale, or decided_by"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> List[CrossMeetingDecisionResponse]:
    """List decisions for the authenticated organization with strict tenant isolation."""
    org_id = str(tenant.organization.id)

    # 1. Validate status filter
    if status_filter:
        s_upper = status_filter.strip().upper()
        if s_upper not in {s.value for s in DecisionStatus}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid decision status '{status_filter}'. Allowed: {[s.value for s in DecisionStatus]}",
            )
        status_filter = s_upper

    # 2. Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    # 3. Validate meeting ownership if specified
    if meeting_id:
        m = db.query(Meeting).filter(Meeting.id == meeting_id, Meeting.organization_id == org_id).first()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    return DecisionLifecycleService.list_decisions(
        db=db,
        organization_id=org_id,
        status=status_filter,
        meeting_id=meeting_id,
        start_date=start_date,
        end_date=end_date,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/decisions/{decision_id}",
    response_model=CrossMeetingDecisionResponse,
    summary="Get decision details and lineage",
    description="Retrieve a single decision by UUID with full ancestor lineage chain.",
)
def get_decision(
    decision_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingDecisionResponse:
    """Retrieve decision details with lineage for the authenticated tenant."""
    org_id = str(tenant.organization.id)
    try:
        return DecisionLifecycleService.get_decision_with_lineage(
            db=db,
            organization_id=org_id,
            decision_id=decision_id,
        )
    except (LineageValidationError, CrossTenantLineageError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Decision not found or access denied.",
        )


@router.post(
    "/decisions/{decision_id}/supersede",
    response_model=CrossMeetingDecisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Explicitly supersede a decision",
    description="Create a new decision in meeting_id that explicitly supersedes target decision_id.",
)
def supersede_decision(
    decision_id: str,
    payload: DecisionSupersedeCreate,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingDecisionResponse:
    """Create a new decision that supersedes target_decision_id within the organization."""
    org_id = str(tenant.organization.id)

    # 1. Verify target decision belongs to active tenant
    target_dec = (
        db.query(Decision)
        .join(Meeting, Decision.meeting_id == Meeting.id)
        .filter(Decision.id == decision_id, Meeting.organization_id == org_id)
        .first()
    )
    if not target_dec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target decision not found or access denied.",
        )

    if target_dec.status in (DecisionStatus.SUPERSEDED.value, DecisionStatus.REVOKED.value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot supersede a decision that is already {target_dec.status}.",
        )

    # 2. Verify new meeting belongs to active tenant
    new_meeting = (
        db.query(Meeting)
        .filter(Meeting.id == payload.meeting_id, Meeting.organization_id == org_id)
        .first()
    )
    if not new_meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target meeting for new decision not found or access denied.",
        )

    # 3. Verify evidence segment if supplied
    if payload.evidence_segment_id:
        seg = (
            db.query(TranscriptSegment)
            .join(Transcript, TranscriptSegment.transcript_id == Transcript.id)
            .join(Meeting, Transcript.meeting_id == Meeting.id)
            .filter(
                TranscriptSegment.id == payload.evidence_segment_id,
                Transcript.meeting_id == payload.meeting_id,
                Meeting.organization_id == org_id,
            )
            .first()
        )
        if not seg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence segment not found in specified meeting or access denied.",
            )

    client_ip = _get_client_ip(request)
    new_dec_data = DecisionCreate(
        description=payload.description.strip(),
        rationale_or_context=payload.rationale_or_context.strip() if payload.rationale_or_context else None,
        decided_by=payload.decided_by.strip() if payload.decided_by else None,
        evidence_segment_id=payload.evidence_segment_id,
        timestamp_seconds=payload.timestamp_seconds,
        status=DecisionStatus.ACTIVE,
    )

    try:
        new_dec = DecisionLifecycleService.supersede_decision_explicit(
            db=db,
            organization_id=org_id,
            target_decision_id=decision_id,
            new_decision_data=new_dec_data,
            meeting_id=payload.meeting_id,
            user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except (CircularLineageError, SelfSupersedingError, LineageValidationError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found or cross-tenant access denied.",
        )

    return DecisionLifecycleService.get_decision_with_lineage(
        db=db,
        organization_id=org_id,
        decision_id=new_dec.id,
    )


@router.post(
    "/decisions/{decision_id}/link-superseded",
    response_model=CrossMeetingDecisionResponse,
    summary="Link existing decision as superseded",
    description="Explicitly link an earlier decision as superseded by decision_id.",
)
def link_superseded_decision(
    decision_id: str,
    payload: DecisionLinkSupersededRequest,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingDecisionResponse:
    """Link an existing earlier decision as superseded by active decision_id."""
    org_id = str(tenant.organization.id)
    target_id = payload.superseded_decision_id.strip()

    if decision_id == target_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A decision cannot supersede itself.",
        )

    # 1. Verify active decision belongs to tenant
    dec_active = (
        db.query(Decision)
        .join(Meeting, Decision.meeting_id == Meeting.id)
        .filter(Decision.id == decision_id, Meeting.organization_id == org_id)
        .first()
    )
    if not dec_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Decision not found or access denied.",
        )

    # 2. Verify target decision to supersede belongs to tenant
    dec_target = (
        db.query(Decision)
        .join(Meeting, Decision.meeting_id == Meeting.id)
        .filter(Decision.id == target_id, Meeting.organization_id == org_id)
        .first()
    )
    if not dec_target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target decision to supersede not found or access denied.",
        )

    client_ip = _get_client_ip(request)
    try:
        DecisionLifecycleService.supersede_existing_decision(
            db=db,
            organization_id=org_id,
            target_decision_id=target_id,
            superseding_decision_id=decision_id,
            user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except (CircularLineageError, SelfSupersedingError, LineageValidationError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found or cross-tenant access denied.",
        )

    return DecisionLifecycleService.get_decision_with_lineage(
        db=db,
        organization_id=org_id,
        decision_id=decision_id,
    )


@router.post(
    "/decisions/{decision_id}/revoke",
    response_model=CrossMeetingDecisionResponse,
    summary="Revoke an active decision with reason",
    description="Revoke an active decision with an immutable audit record and required reason.",
)
def revoke_decision(
    decision_id: str,
    payload: DecisionRevokeRequest,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingDecisionResponse:
    """Revoke an active decision within the tenant organization."""
    org_id = str(tenant.organization.id)
    reason = payload.reason.strip()
    if not reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Revocation reason cannot be empty.",
        )

    # 1. Verify decision belongs to tenant
    dec = (
        db.query(Decision)
        .join(Meeting, Decision.meeting_id == Meeting.id)
        .filter(Decision.id == decision_id, Meeting.organization_id == org_id)
        .first()
    )
    if not dec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Decision not found or access denied.",
        )

    if dec.status == DecisionStatus.REVOKED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decision is already revoked.",
        )

    client_ip = _get_client_ip(request)
    try:
        DecisionLifecycleService.revoke_decision(
            db=db,
            organization_id=org_id,
            decision_id=decision_id,
            reason=reason,
            user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except LineageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Decision not found or access denied.",
        )

    return DecisionLifecycleService.get_decision_with_lineage(
        db=db,
        organization_id=org_id,
        decision_id=decision_id,
    )


# ===========================================================================
# 2. Commitments Endpoints
# ===========================================================================

@router.get(
    "/commitments",
    response_model=List[CrossMeetingCommitmentResponse],
    summary="List commitments across meetings",
    description="List commitments with filters for status, owner, overdue state, deadlines, and meeting.",
)
def list_commitments(
    status_filter: Optional[str] = Query(None, alias="status", description="PENDING, IN_PROGRESS, COMPLETED, CANCELLED"),
    owner_name: Optional[str] = Query(None, description="Search owner name"),
    owner_user_id: Optional[str] = Query(None, description="Filter by owner user UUID"),
    is_overdue: Optional[bool] = Query(None, description="Filter by overdue status"),
    due_before: Optional[datetime] = Query(None, description="Filter commitments due before timestamp"),
    due_after: Optional[datetime] = Query(None, description="Filter commitments due after timestamp"),
    meeting_id: Optional[str] = Query(None, description="Filter by origin meeting UUID"),
    search: Optional[str] = Query(None, description="Text search in description or owner name"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> List[CrossMeetingCommitmentResponse]:
    """List commitments across meetings with dynamic overdue computation."""
    org_id = str(tenant.organization.id)

    # 1. Validate status filter
    if status_filter:
        s_upper = status_filter.strip().upper()
        if s_upper not in {s.value for s in ActionItemStatus}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action item status '{status_filter}'. Allowed: {[s.value for s in ActionItemStatus]}",
            )
        status_filter = s_upper

    # 2. Validate deadline range
    if due_before and due_after and due_after > due_before:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="due_after cannot be later than due_before.",
        )

    # 3. Validate meeting ownership if specified
    if meeting_id:
        m = db.query(Meeting).filter(Meeting.id == meeting_id, Meeting.organization_id == org_id).first()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    return CommitmentTrackingService.list_commitments(
        db=db,
        organization_id=org_id,
        status=status_filter,
        owner_name=owner_name,
        owner_user_id=owner_user_id,
        is_overdue=is_overdue,
        due_before=due_before,
        due_after=due_after,
        meeting_id=meeting_id,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/commitments/{action_item_id}",
    response_model=CrossMeetingCommitmentResponse,
    summary="Get commitment details and review history",
    description="Retrieve commitment details and complete chronological review history.",
)
def get_commitment(
    action_item_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingCommitmentResponse:
    """Retrieve commitment details and review history for the authenticated tenant."""
    org_id = str(tenant.organization.id)
    try:
        return CommitmentTrackingService.get_commitment_with_history(
            db=db,
            organization_id=org_id,
            action_item_id=action_item_id,
        )
    except (LineageValidationError, CrossTenantLineageError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commitment not found or access denied.",
        )


@router.post(
    "/commitments/{action_item_id}/reviews",
    response_model=CrossMeetingCommitmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a cross-meeting commitment review",
    description="Append an immutable review event to a commitment from a review meeting.",
)
def record_commitment_review(
    action_item_id: str,
    payload: CommitmentReviewUpdate,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingCommitmentResponse:
    """Append a review record to an action item and update denormalized metadata."""
    org_id = str(tenant.organization.id)

    # 1. Verify action item belongs to tenant
    ai = (
        db.query(ActionItem)
        .join(Meeting, ActionItem.meeting_id == Meeting.id)
        .filter(ActionItem.id == action_item_id, Meeting.organization_id == org_id)
        .first()
    )
    if not ai:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commitment not found or access denied.",
        )

    # 2. Verify reviewing meeting belongs to tenant
    rev_meeting = (
        db.query(Meeting)
        .filter(Meeting.id == payload.reviewing_meeting_id, Meeting.organization_id == org_id)
        .first()
    )
    if not rev_meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reviewing meeting not found or access denied.",
        )

    # 3. Verify evidence segment if provided
    if payload.evidence_segment_id:
        seg = (
            db.query(TranscriptSegment)
            .join(Transcript, TranscriptSegment.transcript_id == Transcript.id)
            .join(Meeting, Transcript.meeting_id == Meeting.id)
            .filter(
                TranscriptSegment.id == payload.evidence_segment_id,
                Transcript.meeting_id == payload.reviewing_meeting_id,
                Meeting.organization_id == org_id,
            )
            .first()
        )
        if not seg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evidence segment not found in reviewing meeting or access denied.",
            )

    client_ip = _get_client_ip(request)
    try:
        CommitmentTrackingService.review_commitment(
            db=db,
            organization_id=org_id,
            action_item_id=action_item_id,
            reviewing_meeting_id=payload.reviewing_meeting_id,
            evidence_segment_id=payload.evidence_segment_id,
            status=payload.status,
            review_notes=payload.review_notes.strip() if payload.review_notes else None,
            reviewer_user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except LineageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found or cross-tenant access denied.",
        )

    return CommitmentTrackingService.get_commitment_with_history(
        db=db,
        organization_id=org_id,
        action_item_id=action_item_id,
    )


@router.get(
    "/commitments/{action_item_id}/reviews",
    response_model=List[ActionItemReviewResponse],
    summary="List commitment review events",
    description="List all historical review entries recorded for this commitment in chronological order.",
)
def list_commitment_reviews(
    action_item_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> List[ActionItemReviewResponse]:
    """Retrieve review entries for an action item within the organization."""
    org_id = str(tenant.organization.id)
    try:
        commitment = CommitmentTrackingService.get_commitment_with_history(
            db=db,
            organization_id=org_id,
            action_item_id=action_item_id,
        )
        return commitment.reviews
    except (LineageValidationError, CrossTenantLineageError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Commitment not found or access denied.",
        )


# ===========================================================================
# 3. Unresolved Questions Endpoints
# ===========================================================================

@router.get(
    "/questions",
    response_model=List[CrossMeetingQuestionResponse],
    summary="List unresolved questions across organization",
    description="List questions with optional status, meeting, date range, and keyword search filters.",
)
def list_questions(
    status_filter: Optional[str] = Query(None, alias="status", description="OPEN, RESOLVED, or DEFERRED"),
    meeting_id: Optional[str] = Query(None, description="Filter by origin meeting UUID"),
    start_date: Optional[datetime] = Query(None, description="Lower bound on creation date"),
    end_date: Optional[datetime] = Query(None, description="Upper bound on creation date"),
    search: Optional[str] = Query(None, description="Text search in question text, raised_by, or notes"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> List[CrossMeetingQuestionResponse]:
    """List questions across meetings for the authenticated organization."""
    org_id = str(tenant.organization.id)

    # 1. Validate status filter
    if status_filter:
        s_upper = status_filter.strip().upper()
        if s_upper not in {s.value for s in QuestionStatus}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid question status '{status_filter}'. Allowed: {[s.value for s in QuestionStatus]}",
            )
        status_filter = s_upper

    # 2. Validate date range
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    # 3. Validate meeting ownership if specified
    if meeting_id:
        m = db.query(Meeting).filter(Meeting.id == meeting_id, Meeting.organization_id == org_id).first()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    return UnresolvedQuestionService.list_questions(
        db=db,
        organization_id=org_id,
        status=status_filter,
        meeting_id=meeting_id,
        start_date=start_date,
        end_date=end_date,
        search=search,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/questions/{question_id}",
    response_model=CrossMeetingQuestionResponse,
    summary="Get unresolved question details",
    description="Retrieve question details including origin meeting, resolving meeting, and decision links.",
)
def get_question(
    question_id: str,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingQuestionResponse:
    """Retrieve details of a single question for the authenticated tenant."""
    org_id = str(tenant.organization.id)
    try:
        return UnresolvedQuestionService.get_question(
            db=db,
            organization_id=org_id,
            question_id=question_id,
        )
    except (LineageValidationError, CrossTenantLineageError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found or access denied.",
        )


@router.post(
    "/questions/{question_id}/resolve",
    response_model=CrossMeetingQuestionResponse,
    summary="Resolve an unresolved question",
    description="Resolve an open question by linking the resolving meeting, decision, and notes.",
)
def resolve_question(
    question_id: str,
    payload: QuestionResolutionUpdate,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingQuestionResponse:
    """Resolve an open question within the tenant organization."""
    org_id = str(tenant.organization.id)

    # 1. Verify question belongs to tenant
    q = (
        db.query(UnresolvedQuestion)
        .join(Meeting, UnresolvedQuestion.meeting_id == Meeting.id)
        .filter(UnresolvedQuestion.id == question_id, Meeting.organization_id == org_id)
        .first()
    )
    if not q:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found or access denied.",
        )

    # 2. Verify resolving meeting belongs to tenant
    res_meeting = (
        db.query(Meeting)
        .filter(Meeting.id == payload.resolving_meeting_id, Meeting.organization_id == org_id)
        .first()
    )
    if not res_meeting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resolving meeting not found or access denied.",
        )

    # 3. Verify resolving decision if provided
    if payload.resolving_decision_id:
        dec = (
            db.query(Decision)
            .join(Meeting, Decision.meeting_id == Meeting.id)
            .filter(
                Decision.id == payload.resolving_decision_id,
                Meeting.organization_id == org_id,
            )
            .first()
        )
        if not dec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resolving decision not found or access denied.",
            )

    client_ip = _get_client_ip(request)
    try:
        UnresolvedQuestionService.resolve_question(
            db=db,
            organization_id=org_id,
            question_id=question_id,
            resolving_meeting_id=payload.resolving_meeting_id,
            resolving_decision_id=payload.resolving_decision_id,
            resolution_notes=payload.resolution_notes.strip() if payload.resolution_notes else None,
            user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except LineageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found or cross-tenant access denied.",
        )

    return UnresolvedQuestionService.get_question(
        db=db,
        organization_id=org_id,
        question_id=question_id,
    )


@router.post(
    "/questions/{question_id}/reopen",
    response_model=CrossMeetingQuestionResponse,
    summary="Reopen a question back to OPEN status",
    description="Reopen a previously resolved question back to OPEN status with audit logging.",
)
def reopen_question(
    question_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> CrossMeetingQuestionResponse:
    """Reopen a question back to OPEN status."""
    org_id = str(tenant.organization.id)

    # 1. Verify question belongs to tenant
    q = (
        db.query(UnresolvedQuestion)
        .join(Meeting, UnresolvedQuestion.meeting_id == Meeting.id)
        .filter(UnresolvedQuestion.id == question_id, Meeting.organization_id == org_id)
        .first()
    )
    if not q:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found or access denied.",
        )

    client_ip = _get_client_ip(request)
    try:
        UnresolvedQuestionService.reopen_question(
            db=db,
            organization_id=org_id,
            question_id=question_id,
            user_id=tenant.user.id,
            ip_address=client_ip,
        )
    except LineageValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except CrossTenantLineageError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found or access denied.",
        )

    return UnresolvedQuestionService.get_question(
        db=db,
        organization_id=org_id,
        question_id=question_id,
    )


# ===========================================================================
# 4. Conflicts Endpoints
# ===========================================================================

@router.get(
    "/conflicts",
    response_model=List[DecisionConflictItem],
    summary="Detect potential contradictions across active decisions",
    description="Identify potential contradictory stances or mutually exclusive technologies among active decisions.",
)
def get_conflicts(
    subject: Optional[str] = Query(None, description="Optional keyword or subject filter"),
    meeting_id: Optional[str] = Query(None, description="Optional meeting UUID filter"),
    tenant: TenantContext = Depends(get_current_tenant_membership),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(cross_meeting_rate_limiter),
) -> List[DecisionConflictItem]:
    """Detect heuristic conflicts across active decisions for the authenticated organization."""
    org_id = str(tenant.organization.id)

    # 1. Validate meeting ownership if provided
    if meeting_id:
        m = db.query(Meeting).filter(Meeting.id == meeting_id, Meeting.organization_id == org_id).first()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meeting not found or access denied.",
            )

    detected = ConflictDetector.detect_conflicts(
        db=db,
        organization_id=org_id,
        subject_filter=subject.strip() if subject else None,
    )

    if meeting_id:
        # Filter conflict pairs involving the specified meeting
        m_dec_ids = {
            d[0]
            for d in db.query(Decision.id).filter(Decision.meeting_id == meeting_id).all()
        }
        detected = [
            c for c in detected
            if c.decision_a_id in m_dec_ids or c.decision_b_id in m_dec_ids
        ]

    return detected
