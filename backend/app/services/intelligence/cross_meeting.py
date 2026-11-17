"""Cross-meeting intelligence service layer.

Provides organization-scoped services for:
1. Decision Lifecycle (listing, lineage reconstruction, explicit superseding, revocation).
2. Commitment Tracking (listing, overdue computation, cross-meeting review, append-only review history).
3. Unresolved Questions (listing, explicit resolution, reopening, audit logging).

Enforces:
- Strict tenant boundary isolation derived from authenticated context.
- Explicit lifecycle progression (no automated superseding or keyword-based resolution).
- Preservation of origin meeting and evidence grounding.
- Database transaction atomicity and compliance audit logging.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Union, Dict, Any, Tuple
from sqlalchemy import or_, and_, desc, asc, func
from sqlalchemy.orm import Session, aliased

from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    ActionItemReview,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.models.transcription import TranscriptSegment
from app.models.memory import AuditLog
from app.schemas.intelligence import (
    DecisionCreate,
    DecisionResponse,
    CrossMeetingDecisionResponse,
    ActionItemResponse,
    ActionItemReviewResponse,
    CrossMeetingCommitmentResponse,
    UnresolvedQuestionResponse,
    CrossMeetingQuestionResponse,
)
from app.services.intelligence.lineage import (
    validate_decision_lineage,
    supersede_decision,
    record_commitment_review,
    resolve_unresolved_question,
    get_decision_lineage,
    LineageValidationError,
    CrossTenantLineageError,
    CircularLineageError,
    SelfSupersedingError,
)

logger = logging.getLogger(__name__)


def _record_audit(
    db: Session,
    organization_id: str,
    action: str,
    entity_type: str,
    entity_id: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> None:
    """Helper to persist an immutable AuditLog entry within the active transaction."""
    entry = AuditLog(
        organization_id=organization_id,
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        details=json.dumps(details),
        ip_address=ip_address,
    )
    db.add(entry)


# =========================================================================
# 1. Decision Lifecycle Service
# =========================================================================

class DecisionLifecycleService:
    """Organization-scoped service for managing decisions across meetings."""

    @staticmethod
    def list_decisions(
        db: Session,
        organization_id: str,
        status: Optional[Union[DecisionStatus, str]] = None,
        meeting_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CrossMeetingDecisionResponse]:
        """List decisions across meetings for an organization with filtering.

        Args:
            db: Database session.
            organization_id: Authenticated organization UUID.
            status: Optional status filter (ACTIVE, SUPERSEDED, REVOKED).
            meeting_id: Optional filter for a specific meeting.
            start_date: Optional lower bound on creation time.
            end_date: Optional upper bound on creation time.
            search: Optional text search in description or context.
            limit: Max records to return.
            offset: Pagination offset.

        Returns:
            List of CrossMeetingDecisionResponse items.
        """
        effective_date_expr = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at).label("meeting_date")
        query = (
            db.query(Decision, Meeting.title, effective_date_expr)
            .join(Meeting, Decision.meeting_id == Meeting.id)
            .filter(Meeting.organization_id == organization_id)
        )

        if status:
            status_val = status.value if isinstance(status, DecisionStatus) else str(status).upper()
            query = query.filter(Decision.status == status_val)

        if meeting_id:
            query = query.filter(Decision.meeting_id == meeting_id)

        if start_date:
            query = query.filter(effective_date_expr >= start_date)

        if end_date:
            query = query.filter(effective_date_expr <= end_date)

        if search and search.strip():
            raw_search = search.strip()
            term = f"%{raw_search}%"
            words = [w for w in raw_search.split() if len(w) >= 2]
            if len(words) > 1:
                all_words_filter = and_(*[
                    or_(
                        Decision.description.ilike(f"%{w}%"),
                        Decision.rationale_or_context.ilike(f"%{w}%"),
                        Decision.decided_by.ilike(f"%{w}%"),
                    )
                    for w in words
                ])
                query = query.filter(
                    or_(
                        Decision.description.ilike(term),
                        Decision.rationale_or_context.ilike(term),
                        Decision.decided_by.ilike(term),
                        all_words_filter,
                    )
                )
            else:
                query = query.filter(
                    or_(
                        Decision.description.ilike(term),
                        Decision.rationale_or_context.ilike(term),
                        Decision.decided_by.ilike(term),
                    )
                )

        query = query.order_by(Decision.created_at.desc()).offset(offset).limit(limit)
        results = query.all()

        responses: List[CrossMeetingDecisionResponse] = []
        for dec, m_title, m_date in results:
            # Check if this decision was superseded by a later decision
            superseding_dec = (
                db.query(Decision.id)
                .filter(Decision.supersedes_decision_id == dec.id)
                .first()
            )
            superseded_by_id = superseding_dec[0] if superseding_dec else None

            resp = CrossMeetingDecisionResponse(
                id=dec.id,
                meeting_id=dec.meeting_id,
                description=dec.description,
                rationale_or_context=dec.rationale_or_context,
                decided_by=dec.decided_by,
                evidence_segment_id=dec.evidence_segment_id,
                timestamp_seconds=dec.timestamp_seconds,
                status=dec.status,
                supersedes_decision_id=dec.supersedes_decision_id,
                created_at=dec.created_at,
                updated_at=dec.updated_at,
                meeting_title=m_title,
                meeting_date=m_date,
                superseded_by_decision_id=superseded_by_id,
            )
            responses.append(resp)

        return responses

    @staticmethod
    def get_decision_with_lineage(
        db: Session,
        organization_id: str,
        decision_id: str,
    ) -> CrossMeetingDecisionResponse:
        """Retrieve a specific decision and its full historical ancestor lineage.

        Raises:
            LineageValidationError: If decision not found.
            CrossTenantLineageError: If decision belongs to another organization.
        """
        effective_date_expr = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at).label("meeting_date")
        row = (
            db.query(Decision, Meeting.title, effective_date_expr)
            .join(Meeting, Decision.meeting_id == Meeting.id)
            .filter(Decision.id == decision_id)
            .first()
        )
        if not row:
            raise LineageValidationError(f"Decision '{decision_id}' was not found.")

        dec, m_title, m_date = row
        meeting = db.query(Meeting).filter(Meeting.id == dec.meeting_id).first()
        if not meeting or str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Access to decisions from another organization is strictly forbidden.")

        # Reconstruct full ancestor lineage chain
        ancestors = get_decision_lineage(db, decision_id)
        lineage_responses = [DecisionResponse.model_validate(a) for a in ancestors]

        # Check for superseding decision
        superseding_dec = (
            db.query(Decision.id)
            .filter(Decision.supersedes_decision_id == dec.id)
            .first()
        )
        superseded_by_id = superseding_dec[0] if superseding_dec else None

        return CrossMeetingDecisionResponse(
            id=dec.id,
            meeting_id=dec.meeting_id,
            description=dec.description,
            rationale_or_context=dec.rationale_or_context,
            decided_by=dec.decided_by,
            evidence_segment_id=dec.evidence_segment_id,
            timestamp_seconds=dec.timestamp_seconds,
            status=dec.status,
            supersedes_decision_id=dec.supersedes_decision_id,
            created_at=dec.created_at,
            updated_at=dec.updated_at,
            meeting_title=m_title,
            meeting_date=m_date,
            superseded_by_decision_id=superseded_by_id,
            lineage_depth=len(lineage_responses),
            lineage=lineage_responses,
        )

    @staticmethod
    def supersede_decision_explicit(
        db: Session,
        organization_id: str,
        target_decision_id: str,
        new_decision_data: DecisionCreate,
        meeting_id: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Decision:
        """Explicitly create a new decision in meeting_id that supersedes target_decision_id.

        NEVER performed automatically; must be explicitly requested.
        """
        # 1. Validate new meeting belongs to tenant
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting or str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Meeting does not belong to the active organization.")

        # 2. Validate lineage between target decision and new meeting
        target_dec = validate_decision_lineage(
            db=db,
            target_decision_id=target_decision_id,
            new_meeting_id=meeting_id,
        )

        # 3. Create new decision
        new_dec = Decision(
            meeting_id=meeting_id,
            description=new_decision_data.description,
            rationale_or_context=new_decision_data.rationale_or_context,
            decided_by=new_decision_data.decided_by,
            evidence_segment_id=new_decision_data.evidence_segment_id,
            timestamp_seconds=new_decision_data.timestamp_seconds,
            status=DecisionStatus.ACTIVE.value,
            supersedes_decision_id=target_dec.id,
        )
        db.add(new_dec)
        db.flush()

        # 4. Mark target decision as SUPERSEDED
        old_status = target_dec.status
        target_dec.status = DecisionStatus.SUPERSEDED.value

        # 5. Persist audit log
        _record_audit(
            db=db,
            organization_id=organization_id,
            action="SUPERSEDE_DECISION",
            entity_type="Decision",
            entity_id=new_dec.id,
            details={
                "superseded_decision_id": target_dec.id,
                "superseded_previous_status": old_status,
                "new_decision_id": new_dec.id,
                "meeting_id": meeting_id,
            },
            user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(new_dec)
        return new_dec

    @staticmethod
    def supersede_existing_decision(
        db: Session,
        organization_id: str,
        target_decision_id: str,
        superseding_decision_id: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Decision:
        """Explicitly link two existing decisions within the organization as target -> superseded."""
        superseding_dec = db.query(Decision).filter(Decision.id == superseding_decision_id).first()
        if not superseding_dec:
            raise LineageValidationError(f"Superseding decision '{superseding_decision_id}' was not found.")

        superseding_meeting = db.query(Meeting).filter(Meeting.id == superseding_dec.meeting_id).first()
        if not superseding_meeting or str(superseding_meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Superseding decision belongs to another organization.")

        updated_dec = supersede_decision(
            db=db,
            superseding_decision=superseding_dec,
            target_decision_id=target_decision_id,
            mark_previous_superseded=True,
        )

        _record_audit(
            db=db,
            organization_id=organization_id,
            action="LINK_SUPERSEDING_DECISION",
            entity_type="Decision",
            entity_id=superseding_dec.id,
            details={
                "target_decision_id": target_decision_id,
                "superseding_decision_id": superseding_dec.id,
            },
            user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(updated_dec)
        return updated_dec

    @staticmethod
    def revoke_decision(
        db: Session,
        organization_id: str,
        decision_id: str,
        reason: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Decision:
        """Revoke an active or superseded decision with optional reason."""
        dec = db.query(Decision).filter(Decision.id == decision_id).first()
        if not dec:
            raise LineageValidationError(f"Decision '{decision_id}' was not found.")

        meeting = db.query(Meeting).filter(Meeting.id == dec.meeting_id).first()
        if not meeting or str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Decision belongs to another organization.")

        old_status = dec.status
        dec.status = DecisionStatus.REVOKED.value
        if reason:
            rev_note = f"\n[REVOKED: {reason.strip()}]"
            dec.rationale_or_context = (dec.rationale_or_context or "") + rev_note

        _record_audit(
            db=db,
            organization_id=organization_id,
            action="REVOKE_DECISION",
            entity_type="Decision",
            entity_id=dec.id,
            details={
                "old_status": old_status,
                "new_status": DecisionStatus.REVOKED.value,
                "reason": reason,
            },
            user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(dec)
        return dec


# =========================================================================
# 2. Commitment Tracking Service
# =========================================================================

class CommitmentTrackingService:
    """Organization-scoped service for tracking commitments (ActionItems) and reviews across meetings."""

    @staticmethod
    def list_commitments(
        db: Session,
        organization_id: str,
        status: Optional[Union[ActionItemStatus, str]] = None,
        owner_name: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        is_overdue: Optional[bool] = None,
        due_before: Optional[datetime] = None,
        due_after: Optional[datetime] = None,
        meeting_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CrossMeetingCommitmentResponse]:
        """List commitments across meetings with filtering and dynamic overdue computation."""
        ReviewMeeting = aliased(Meeting)

        effective_origin_date = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at).label("origin_meeting_date")
        query = (
            db.query(
                ActionItem,
                Meeting.title.label("origin_meeting_title"),
                effective_origin_date,
                ReviewMeeting.title.label("last_reviewed_meeting_title"),
            )
            .join(Meeting, ActionItem.meeting_id == Meeting.id)
            .outerjoin(ReviewMeeting, ActionItem.last_reviewed_meeting_id == ReviewMeeting.id)
            .filter(Meeting.organization_id == organization_id)
        )

        if status:
            status_val = status.value if isinstance(status, ActionItemStatus) else str(status).upper()
            query = query.filter(ActionItem.status == status_val)

        if owner_name and owner_name.strip():
            query = query.filter(ActionItem.owner_name.ilike(f"%{owner_name.strip()}%"))

        if owner_user_id:
            query = query.filter(ActionItem.owner_user_id == owner_user_id)

        if meeting_id:
            query = query.filter(ActionItem.meeting_id == meeting_id)

        if due_before:
            query = query.filter(ActionItem.deadline <= due_before)

        if due_after:
            query = query.filter(ActionItem.deadline >= due_after)

        if search and search.strip():
            raw_search = search.strip()
            term = f"%{raw_search}%"
            words = [w for w in raw_search.split() if len(w) >= 2]
            if len(words) > 1:
                all_words_filter = and_(*[
                    or_(
                        ActionItem.description.ilike(f"%{w}%"),
                        ActionItem.owner_name.ilike(f"%{w}%"),
                    )
                    for w in words
                ])
                query = query.filter(
                    or_(
                        ActionItem.description.ilike(term),
                        ActionItem.owner_name.ilike(term),
                        all_words_filter,
                    )
                )
            else:
                query = query.filter(
                    or_(
                        ActionItem.description.ilike(term),
                        ActionItem.owner_name.ilike(term),
                    )
                )

        now = datetime.now(timezone.utc)
        if is_overdue is True:
            query = query.filter(
                ActionItem.deadline.isnot(None),
                ActionItem.deadline < now,
                ActionItem.status.notin_([ActionItemStatus.COMPLETED.value, ActionItemStatus.CANCELLED.value]),
            )
        elif is_overdue is False:
            query = query.filter(
                or_(
                    ActionItem.deadline.is_(None),
                    ActionItem.deadline >= now,
                    ActionItem.status.in_([ActionItemStatus.COMPLETED.value, ActionItemStatus.CANCELLED.value]),
                )
            )

        query = query.order_by(
            ActionItem.deadline.asc().nullslast(),
            ActionItem.created_at.desc()
        ).offset(offset).limit(limit)

        results = query.all()

        responses: List[CrossMeetingCommitmentResponse] = []
        for ai, orig_title, orig_date, rev_title in results:
            # Compute dynamic overdue flag
            overdue = bool(
                ai.deadline is not None
                and (ai.deadline.tzinfo is not None and ai.deadline < now or ai.deadline.replace(tzinfo=timezone.utc) < now)
                and ai.status not in (ActionItemStatus.COMPLETED.value, ActionItemStatus.CANCELLED.value)
            )

            # Count total reviews
            review_count = (
                db.query(ActionItemReview.id)
                .filter(ActionItemReview.action_item_id == ai.id)
                .count()
            )

            resp = CrossMeetingCommitmentResponse(
                id=ai.id,
                meeting_id=ai.meeting_id,
                description=ai.description,
                owner_name=ai.owner_name,
                owner_user_id=ai.owner_user_id,
                deadline=ai.deadline,
                status=ai.status,
                evidence_segment_id=ai.evidence_segment_id,
                timestamp_seconds=ai.timestamp_seconds,
                last_reviewed_meeting_id=ai.last_reviewed_meeting_id,
                last_evidence_segment_id=ai.last_evidence_segment_id,
                last_reviewed_at=ai.last_reviewed_at,
                created_at=ai.created_at,
                updated_at=ai.updated_at,
                origin_meeting_title=orig_title,
                origin_meeting_date=orig_date,
                last_reviewed_meeting_title=rev_title,
                is_overdue=overdue,
                review_count=review_count,
            )
            responses.append(resp)

        return responses

    @staticmethod
    def get_commitment_with_history(
        db: Session,
        organization_id: str,
        action_item_id: str,
    ) -> CrossMeetingCommitmentResponse:
        """Retrieve a specific action item with its complete chronological review history."""
        row = (
            db.query(ActionItem, Meeting)
            .join(Meeting, ActionItem.meeting_id == Meeting.id)
            .filter(ActionItem.id == action_item_id)
            .first()
        )
        if not row:
            raise LineageValidationError(f"Action item '{action_item_id}' was not found.")

        ai, meeting = row
        if str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Action item belongs to another organization.")

        # Review history
        reviews_query = (
            db.query(ActionItemReview, Meeting.title, TranscriptSegment.text)
            .join(Meeting, ActionItemReview.meeting_id == Meeting.id)
            .outerjoin(TranscriptSegment, ActionItemReview.evidence_segment_id == TranscriptSegment.id)
            .filter(ActionItemReview.action_item_id == ai.id)
            .order_by(ActionItemReview.reviewed_at.asc())
            .all()
        )

        review_responses: List[ActionItemReviewResponse] = []
        for rev, m_title, seg_text in reviews_query:
            review_responses.append(
                ActionItemReviewResponse(
                    id=rev.id,
                    action_item_id=rev.action_item_id,
                    meeting_id=rev.meeting_id,
                    meeting_title=m_title,
                    evidence_segment_id=rev.evidence_segment_id,
                    evidence_quote=seg_text,
                    reviewer_user_id=rev.reviewer_user_id,
                    status_at_review=rev.status_at_review,
                    review_notes=rev.review_notes,
                    reviewed_at=rev.reviewed_at,
                    created_at=rev.created_at,
                )
            )

        rev_meeting_title = None
        if ai.last_reviewed_meeting_id:
            rev_m = db.query(Meeting.title).filter(Meeting.id == ai.last_reviewed_meeting_id).first()
            rev_meeting_title = rev_m[0] if rev_m else None

        now = datetime.now(timezone.utc)
        overdue = bool(
            ai.deadline is not None
            and (ai.deadline.tzinfo is not None and ai.deadline < now or ai.deadline.replace(tzinfo=timezone.utc) < now)
            and ai.status not in (ActionItemStatus.COMPLETED.value, ActionItemStatus.CANCELLED.value)
        )

        return CrossMeetingCommitmentResponse(
            id=ai.id,
            meeting_id=ai.meeting_id,
            description=ai.description,
            owner_name=ai.owner_name,
            owner_user_id=ai.owner_user_id,
            deadline=ai.deadline,
            status=ai.status,
            evidence_segment_id=ai.evidence_segment_id,
            timestamp_seconds=ai.timestamp_seconds,
            last_reviewed_meeting_id=ai.last_reviewed_meeting_id,
            last_evidence_segment_id=ai.last_evidence_segment_id,
            last_reviewed_at=ai.last_reviewed_at,
            created_at=ai.created_at,
            updated_at=ai.updated_at,
            origin_meeting_title=meeting.title,
            origin_meeting_date=meeting.actual_start_time or meeting.scheduled_start_time or meeting.created_at,
            last_reviewed_meeting_title=rev_meeting_title,
            is_overdue=overdue,
            review_count=len(review_responses),
            reviews=review_responses,
        )

    @staticmethod
    def review_commitment(
        db: Session,
        organization_id: str,
        action_item_id: str,
        reviewing_meeting_id: str,
        evidence_segment_id: Optional[str] = None,
        status: Optional[Union[ActionItemStatus, str]] = None,
        review_notes: Optional[str] = None,
        reviewer_user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> ActionItem:
        """Record a cross-meeting review event, updating continuity cache and appending history."""
        # 1. Validate reviewing meeting belongs to organization
        reviewing_meeting = db.query(Meeting).filter(Meeting.id == reviewing_meeting_id).first()
        if not reviewing_meeting or str(reviewing_meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Reviewing meeting does not belong to the active organization.")

        # 2. Record review using lineage module
        updated_item = record_commitment_review(
            db=db,
            action_item_id=action_item_id,
            reviewing_meeting_id=reviewing_meeting_id,
            evidence_segment_id=evidence_segment_id,
            status=status,
            review_notes=review_notes,
            reviewer_user_id=reviewer_user_id,
        )

        # 3. Audit trail
        _record_audit(
            db=db,
            organization_id=organization_id,
            action="REVIEW_COMMITMENT",
            entity_type="ActionItem",
            entity_id=updated_item.id,
            details={
                "reviewing_meeting_id": reviewing_meeting_id,
                "status": updated_item.status,
                "notes": review_notes,
                "evidence_segment_id": evidence_segment_id,
            },
            user_id=reviewer_user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(updated_item)
        return updated_item


# =========================================================================
# 3. Unresolved Question Service
# =========================================================================

class UnresolvedQuestionService:
    """Organization-scoped service for tracking and resolving open questions across meetings."""

    @staticmethod
    def list_questions(
        db: Session,
        organization_id: str,
        status: Optional[Union[QuestionStatus, str]] = None,
        meeting_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CrossMeetingQuestionResponse]:
        """List questions across meetings for an organization with filtering."""
        ResolvingMeeting = aliased(Meeting)
        ResolvingDecision = aliased(Decision)

        effective_origin_date = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at).label("origin_meeting_date")
        query = (
            db.query(
                UnresolvedQuestion,
                Meeting.title.label("origin_meeting_title"),
                effective_origin_date,
                ResolvingMeeting.title.label("resolving_meeting_title"),
                ResolvingDecision.description.label("resolving_decision_description"),
            )
            .join(Meeting, UnresolvedQuestion.meeting_id == Meeting.id)
            .outerjoin(ResolvingMeeting, UnresolvedQuestion.resolving_meeting_id == ResolvingMeeting.id)
            .outerjoin(ResolvingDecision, UnresolvedQuestion.resolving_decision_id == ResolvingDecision.id)
            .filter(Meeting.organization_id == organization_id)
        )

        if status:
            status_val = status.value if isinstance(status, QuestionStatus) else str(status).upper()
            query = query.filter(UnresolvedQuestion.status == status_val)

        if meeting_id:
            query = query.filter(UnresolvedQuestion.meeting_id == meeting_id)

        if start_date:
            query = query.filter(UnresolvedQuestion.created_at >= start_date)

        if end_date:
            query = query.filter(UnresolvedQuestion.created_at <= end_date)

        if search and search.strip():
            raw_search = search.strip()
            term = f"%{raw_search}%"
            words = [w for w in raw_search.split() if len(w) >= 2]
            if len(words) > 1:
                all_words_filter = and_(*[
                    or_(
                        UnresolvedQuestion.question_text.ilike(f"%{w}%"),
                        UnresolvedQuestion.raised_by.ilike(f"%{w}%"),
                        UnresolvedQuestion.resolution_notes.ilike(f"%{w}%"),
                    )
                    for w in words
                ])
                query = query.filter(
                    or_(
                        UnresolvedQuestion.question_text.ilike(term),
                        UnresolvedQuestion.raised_by.ilike(term),
                        UnresolvedQuestion.resolution_notes.ilike(term),
                        all_words_filter,
                    )
                )
            else:
                query = query.filter(
                    or_(
                        UnresolvedQuestion.question_text.ilike(term),
                        UnresolvedQuestion.raised_by.ilike(term),
                        UnresolvedQuestion.resolution_notes.ilike(term),
                    )
                )

        query = query.order_by(
            UnresolvedQuestion.status.asc(),
            UnresolvedQuestion.created_at.desc()
        ).offset(offset).limit(limit)

        results = query.all()

        responses: List[CrossMeetingQuestionResponse] = []
        for q, orig_title, orig_date, res_m_title, res_dec_desc in results:
            resp = CrossMeetingQuestionResponse(
                id=q.id,
                meeting_id=q.meeting_id,
                question_text=q.question_text,
                raised_by=q.raised_by,
                status=q.status,
                evidence_segment_id=q.evidence_segment_id,
                resolved_at=q.resolved_at,
                resolving_meeting_id=q.resolving_meeting_id,
                resolving_decision_id=q.resolving_decision_id,
                resolution_notes=q.resolution_notes,
                created_at=q.created_at,
                updated_at=q.updated_at,
                origin_meeting_title=orig_title,
                origin_meeting_date=orig_date,
                resolving_meeting_title=res_m_title,
                resolving_decision_description=res_dec_desc,
            )
            responses.append(resp)

        return responses

    @staticmethod
    def get_question(
        db: Session,
        organization_id: str,
        question_id: str,
    ) -> CrossMeetingQuestionResponse:
        """Retrieve details of a single unresolved question with meeting and decision context."""
        ResolvingMeeting = aliased(Meeting)
        ResolvingDecision = aliased(Decision)

        effective_origin_date = func.coalesce(Meeting.actual_start_time, Meeting.scheduled_start_time, Meeting.created_at).label("origin_meeting_date")
        row = (
            db.query(
                UnresolvedQuestion,
                Meeting.title.label("origin_meeting_title"),
                effective_origin_date,
                ResolvingMeeting.title.label("resolving_meeting_title"),
                ResolvingDecision.description.label("resolving_decision_description"),
            )
            .join(Meeting, UnresolvedQuestion.meeting_id == Meeting.id)
            .outerjoin(ResolvingMeeting, UnresolvedQuestion.resolving_meeting_id == ResolvingMeeting.id)
            .outerjoin(ResolvingDecision, UnresolvedQuestion.resolving_decision_id == ResolvingDecision.id)
            .filter(UnresolvedQuestion.id == question_id)
            .first()
        )
        if not row:
            raise LineageValidationError(f"Question '{question_id}' was not found.")

        q, orig_title, orig_date, res_m_title, res_dec_desc = row
        meeting = db.query(Meeting).filter(Meeting.id == q.meeting_id).first()
        if not meeting or str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Question belongs to another organization.")

        return CrossMeetingQuestionResponse(
            id=q.id,
            meeting_id=q.meeting_id,
            question_text=q.question_text,
            raised_by=q.raised_by,
            status=q.status,
            evidence_segment_id=q.evidence_segment_id,
            resolved_at=q.resolved_at,
            resolving_meeting_id=q.resolving_meeting_id,
            resolving_decision_id=q.resolving_decision_id,
            resolution_notes=q.resolution_notes,
            created_at=q.created_at,
            updated_at=q.updated_at,
            origin_meeting_title=orig_title,
            origin_meeting_date=orig_date,
            resolving_meeting_title=res_m_title,
            resolving_decision_description=res_dec_desc,
        )

    @staticmethod
    def resolve_question(
        db: Session,
        organization_id: str,
        question_id: str,
        resolving_meeting_id: str,
        resolving_decision_id: Optional[str] = None,
        resolution_notes: Optional[str] = None,
        status: Union[QuestionStatus, str] = QuestionStatus.RESOLVED,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> UnresolvedQuestion:
        """Explicitly resolve a question, linking resolving meeting and optional decision."""
        # 1. Validate resolving meeting belongs to active organization
        res_meeting = db.query(Meeting).filter(Meeting.id == resolving_meeting_id).first()
        if not res_meeting or str(res_meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Resolving meeting belongs to another organization.")

        # 2. Resolve question via lineage module
        resolved_q = resolve_unresolved_question(
            db=db,
            question_id=question_id,
            resolving_meeting_id=resolving_meeting_id,
            resolving_decision_id=resolving_decision_id,
            resolution_notes=resolution_notes,
            status=status,
        )

        # 3. Audit trail
        _record_audit(
            db=db,
            organization_id=organization_id,
            action="RESOLVE_QUESTION",
            entity_type="UnresolvedQuestion",
            entity_id=resolved_q.id,
            details={
                "resolving_meeting_id": resolving_meeting_id,
                "resolving_decision_id": resolving_decision_id,
                "status": resolved_q.status,
                "notes": resolution_notes,
            },
            user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(resolved_q)
        return resolved_q

    @staticmethod
    def reopen_question(
        db: Session,
        organization_id: str,
        question_id: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> UnresolvedQuestion:
        """Reopen an unresolved question back to OPEN status."""
        q = db.query(UnresolvedQuestion).filter(UnresolvedQuestion.id == question_id).first()
        if not q:
            raise LineageValidationError(f"Question '{question_id}' was not found.")

        meeting = db.query(Meeting).filter(Meeting.id == q.meeting_id).first()
        if not meeting or str(meeting.organization_id) != str(organization_id):
            raise CrossTenantLineageError("Question belongs to another organization.")

        old_status = q.status
        q.status = QuestionStatus.OPEN.value
        q.resolved_at = None
        q.resolving_meeting_id = None
        q.resolving_decision_id = None
        if q.resolution_notes:
            q.resolution_notes = f"[REOPENED] Previously: {q.resolution_notes}"

        _record_audit(
            db=db,
            organization_id=organization_id,
            action="REOPEN_QUESTION",
            entity_type="UnresolvedQuestion",
            entity_id=q.id,
            details={
                "old_status": old_status,
                "new_status": QuestionStatus.OPEN.value,
            },
            user_id=user_id,
            ip_address=ip_address,
        )
        db.commit()
        db.refresh(q)
        return q
