"""Decision lineage, commitment continuity, and question resolution service.

Enforces:
- Tenant isolation across cross-meeting operations
- Self-referencing prevention
- Acyclic decision lineage graph (cycle detection)
- State progression for superseding decisions, reviewed commitments, and resolved questions
"""

from datetime import datetime, timezone
from typing import Optional, List, Union, Set
from sqlalchemy.orm import Session

from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    ActionItemReview,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.models.meeting import Meeting
from app.models.transcription import TranscriptSegment


class LineageValidationError(ValueError):
    """Base exception for lineage and cross-meeting integrity validation failures."""
    pass


class CrossTenantLineageError(LineageValidationError):
    """Raised when an operation attempts to link entities across different organizations."""
    pass


class CircularLineageError(LineageValidationError):
    """Raised when an operation would produce a circular dependency in decision lineage."""
    pass


class SelfSupersedingError(LineageValidationError):
    """Raised when a decision attempts to supersede itself."""
    pass


def validate_decision_lineage(
    db: Session,
    target_decision_id: str,
    new_meeting_id: str,
    superseding_decision_id: Optional[str] = None,
) -> Decision:
    """Validate that target_decision can be superseded by a decision in new_meeting_id.

    Args:
        db: Active database session.
        target_decision_id: The UUID of the existing decision to be superseded.
        new_meeting_id: The UUID of the meeting where the new decision is being made.
        superseding_decision_id: Optional UUID of the superseding decision (if already exists/persisted).

    Returns:
        The target Decision instance to be superseded.

    Raises:
        SelfSupersedingError: If superseding_decision_id == target_decision_id.
        LineageValidationError: If either record does not exist.
        CrossTenantLineageError: If the meetings belong to different organizations.
        CircularLineageError: If superseding would introduce a cycle.
    """
    if superseding_decision_id and superseding_decision_id == target_decision_id:
        raise SelfSupersedingError("A decision cannot supersede itself.")

    # 1. Fetch target decision and its meeting
    target_decision = db.query(Decision).filter(Decision.id == target_decision_id).first()
    if not target_decision:
        raise LineageValidationError(f"Target decision to supersede '{target_decision_id}' was not found.")

    target_meeting = db.query(Meeting).filter(Meeting.id == target_decision.meeting_id).first()
    if not target_meeting:
        raise LineageValidationError(f"Meeting for target decision '{target_decision.meeting_id}' was not found.")

    # 2. Fetch new meeting
    new_meeting = db.query(Meeting).filter(Meeting.id == new_meeting_id).first()
    if not new_meeting:
        raise LineageValidationError(f"New meeting '{new_meeting_id}' was not found.")

    # 3. Verify tenant boundary
    if str(target_meeting.organization_id) != str(new_meeting.organization_id):
        raise CrossTenantLineageError(
            f"Cross-organization decision superseding is strictly prohibited. "
            f"Target decision org: {target_meeting.organization_id}, new decision org: {new_meeting.organization_id}."
        )

    # 4. Cycle detection (traverse target's ancestors to ensure superseding_decision_id is not already an ancestor)
    if superseding_decision_id:
        visited: Set[str] = {superseding_decision_id}
        current_id: Optional[str] = target_decision.supersedes_decision_id
        depth = 0
        max_depth = 100

        while current_id and depth < max_depth:
            if current_id in visited:
                raise CircularLineageError(
                    f"Circular decision lineage detected: decision '{superseding_decision_id}' is already in lineage path."
                )
            visited.add(current_id)
            ancestor = db.query(Decision.supersedes_decision_id).filter(Decision.id == current_id).first()
            current_id = ancestor[0] if ancestor else None
            depth += 1

        if depth >= max_depth:
            raise CircularLineageError("Decision lineage chain exceeded maximum traversal depth (potential cycle).")

    return target_decision


def supersede_decision(
    db: Session,
    superseding_decision: Decision,
    target_decision_id: str,
    mark_previous_superseded: bool = True,
) -> Decision:
    """Explicitly link superseding_decision to supersede target_decision_id.

    Validates tenant isolation, self-referencing constraints, and acyclicity.
    Marks target decision as SUPERSEDED and links supersedes_decision_id.

    Args:
        db: Active database session.
        superseding_decision: The Decision model instance that supersedes the earlier decision.
        target_decision_id: The UUID of the earlier decision being superseded.
        mark_previous_superseded: Whether to automatically set target_decision.status = SUPERSEDED.

    Returns:
        The updated superseding_decision instance.
    """
    target_decision = validate_decision_lineage(
        db=db,
        target_decision_id=target_decision_id,
        new_meeting_id=superseding_decision.meeting_id,
        superseding_decision_id=superseding_decision.id,
    )

    if mark_previous_superseded:
        target_decision.status = DecisionStatus.SUPERSEDED.value

    superseding_decision.supersedes_decision_id = target_decision.id
    superseding_decision.status = DecisionStatus.ACTIVE.value

    return superseding_decision


def record_commitment_review(
    db: Session,
    action_item_id: str,
    reviewing_meeting_id: str,
    evidence_segment_id: Optional[str] = None,
    status: Optional[Union[ActionItemStatus, str]] = None,
    review_notes: Optional[str] = None,
    reviewer_user_id: Optional[str] = None,
) -> ActionItem:
    """Record a cross-meeting review or status update for an existing commitment (ActionItem).

    Preserves origin meeting, original evidence segment, and origin timestamp while
    updating continuity review tracking and appending an immutable ActionItemReview history entry.

    Args:
        db: Active database session.
        action_item_id: UUID of the action item being reviewed.
        reviewing_meeting_id: UUID of the meeting where this item was reviewed/discussed.
        evidence_segment_id: Optional UUID of the transcript segment discussing this item in the review meeting.
        status: Optional updated status.
        review_notes: Optional context or discussion summary.
        reviewer_user_id: Optional UUID of the user logging the review.

    Returns:
        The updated ActionItem instance.

    Raises:
        LineageValidationError: If action item or meeting is not found.
        CrossTenantLineageError: If reviewing meeting belongs to a different organization.
    """
    action_item = db.query(ActionItem).filter(ActionItem.id == action_item_id).first()
    if not action_item:
        raise LineageValidationError(f"Action item '{action_item_id}' was not found.")

    origin_meeting = db.query(Meeting).filter(Meeting.id == action_item.meeting_id).first()
    if not origin_meeting:
        raise LineageValidationError(f"Origin meeting '{action_item.meeting_id}' was not found.")

    reviewing_meeting = db.query(Meeting).filter(Meeting.id == reviewing_meeting_id).first()
    if not reviewing_meeting:
        raise LineageValidationError(f"Reviewing meeting '{reviewing_meeting_id}' was not found.")

    if str(origin_meeting.organization_id) != str(reviewing_meeting.organization_id):
        raise CrossTenantLineageError(
            f"Cross-organization commitment review is strictly prohibited. "
            f"Action item org: {origin_meeting.organization_id}, reviewing meeting org: {reviewing_meeting.organization_id}."
        )

    if evidence_segment_id:
        evidence_seg = db.query(TranscriptSegment).filter(TranscriptSegment.id == evidence_segment_id).first()
        if not evidence_seg:
            raise LineageValidationError(f"Evidence transcript segment '{evidence_segment_id}' was not found.")
        action_item.last_evidence_segment_id = evidence_seg.id

    now = datetime.now(timezone.utc)
    action_item.last_reviewed_meeting_id = reviewing_meeting.id
    action_item.last_reviewed_at = now

    if status is not None:
        if isinstance(status, ActionItemStatus):
            action_item.status = status.value
        elif isinstance(status, str):
            action_item.status = ActionItemStatus(status.strip().upper()).value

    # Append immutable review history entry
    review_record = ActionItemReview(
        action_item_id=action_item.id,
        meeting_id=reviewing_meeting.id,
        evidence_segment_id=evidence_segment_id,
        reviewer_user_id=reviewer_user_id,
        status_at_review=action_item.status,
        review_notes=review_notes,
        reviewed_at=now,
    )
    db.add(review_record)

    return action_item


def resolve_unresolved_question(
    db: Session,
    question_id: str,
    resolving_meeting_id: str,
    resolving_decision_id: Optional[str] = None,
    resolution_notes: Optional[str] = None,
    status: Union[QuestionStatus, str] = QuestionStatus.RESOLVED,
) -> UnresolvedQuestion:
    """Record the resolution of an open or deferred question across meetings.

    Enforces tenant boundaries between the original question, the resolving meeting,
    and the resolving decision.

    Args:
        db: Active database session.
        question_id: UUID of the unresolved question to resolve.
        resolving_meeting_id: UUID of the meeting where resolution occurred.
        resolving_decision_id: Optional UUID of the decision that resolved the question.
        resolution_notes: Optional textual explanation of how it was resolved.
        status: Status to assign (defaults to RESOLVED).

    Returns:
        The updated UnresolvedQuestion instance.

    Raises:
        LineageValidationError: If question or meeting not found.
        CrossTenantLineageError: If resolving meeting or decision belongs to another organization.
    """
    question = db.query(UnresolvedQuestion).filter(UnresolvedQuestion.id == question_id).first()
    if not question:
        raise LineageValidationError(f"Unresolved question '{question_id}' was not found.")

    origin_meeting = db.query(Meeting).filter(Meeting.id == question.meeting_id).first()
    if not origin_meeting:
        raise LineageValidationError(f"Origin meeting '{question.meeting_id}' was not found.")

    resolving_meeting = db.query(Meeting).filter(Meeting.id == resolving_meeting_id).first()
    if not resolving_meeting:
        raise LineageValidationError(f"Resolving meeting '{resolving_meeting_id}' was not found.")

    if str(origin_meeting.organization_id) != str(resolving_meeting.organization_id):
        raise CrossTenantLineageError(
            f"Cross-organization question resolution is strictly prohibited. "
            f"Question org: {origin_meeting.organization_id}, resolving meeting org: {resolving_meeting.organization_id}."
        )

    if resolving_decision_id:
        decision = db.query(Decision).filter(Decision.id == resolving_decision_id).first()
        if not decision:
            raise LineageValidationError(f"Resolving decision '{resolving_decision_id}' was not found.")

        decision_meeting = db.query(Meeting).filter(Meeting.id == decision.meeting_id).first()
        if not decision_meeting or str(decision_meeting.organization_id) != str(origin_meeting.organization_id):
            raise CrossTenantLineageError(
                f"Resolving decision '{resolving_decision_id}' belongs to a different organization."
            )
        question.resolving_decision_id = decision.id

    if isinstance(status, QuestionStatus):
        question.status = status.value
    elif isinstance(status, str):
        question.status = QuestionStatus(status.strip().upper()).value

    question.resolving_meeting_id = resolving_meeting.id
    question.resolved_at = datetime.now(timezone.utc)
    if resolution_notes:
        question.resolution_notes = resolution_notes

    return question


def get_decision_lineage(db: Session, decision_id: str) -> List[Decision]:
    """Retrieve the entire ancestor lineage chain of a decision up to the original root.

    Returns the list starting from the root decision down to the given decision.
    """
    chain: List[Decision] = []
    current = db.query(Decision).filter(Decision.id == decision_id).first()
    visited: Set[str] = set()

    while current and current.id not in visited:
        visited.add(current.id)
        chain.append(current)
        if current.supersedes_decision_id:
            current = db.query(Decision).filter(Decision.id == current.supersedes_decision_id).first()
        else:
            break

    chain.reverse()
    return chain
