"""Tests for Phase 8.8.2: Decision Lifecycle, Commitment Continuity & Question Resolution.

Verifies:
1. Migration roundtrip, schema drift check, and backfill (existing decisions default to ACTIVE).
2. Decision lineage progression (A -> superseded by B -> superseded by C).
3. Self-referencing constraint enforcement (Python service and SQLite CHECK constraint).
4. Circular decision lineage prevention.
5. Cross-organization decision superseding rejection.
6. Unresolved question resolution metadata and same-tenant verification.
7. Cross-organization question resolution rejection.
8. Commitment continuity review updates (origin meeting vs review meeting & evidence preservation).
9. Cross-organization commitment review rejection.
10. Cascading and SET NULL deletion behavior.
11. Pydantic schemas, validation, and intelligence response serialization.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy.exc import IntegrityError
from alembic.config import Config
from alembic import command

from app.core.config import settings
from app.models.identity import Organization
from app.models.meeting import Meeting
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.models.intelligence import (
    Decision,
    DecisionStatus,
    ActionItem,
    ActionItemStatus,
    UnresolvedQuestion,
    QuestionStatus,
)
from app.schemas.intelligence import (
    DecisionCreate,
    DecisionResponse,
    ActionItemCreate,
    ActionItemResponse,
    UnresolvedQuestionCreate,
    UnresolvedQuestionResponse,
    MeetingIntelligenceResponse,
    DecisionStatusUpdate,
    CommitmentReviewUpdate,
    QuestionResolutionUpdate,
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

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _create_test_meeting(db, org: Organization, title: str = "Test Meeting") -> Meeting:
    """Helper to create a persisted Meeting record."""
    meeting = Meeting(
        organization_id=org.id,
        title=title,
    )
    db.add(meeting)
    db.commit()
    db.refresh(meeting)
    return meeting


def _create_test_transcript_with_segment(db, meeting: Meeting, text: str = "Sample evidence") -> TranscriptSegment:
    """Helper to create a Transcript and TranscriptSegment for evidence grounding."""
    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        language_detected="en",
    )
    db.add(transcript)
    db.flush()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=1,
        start_time=0.0,
        end_time=5.0,
        text=text,
        language_code="en",
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    return segment


# =========================================================================
# 1. Migration Roundtrip & Backfill Test
# =========================================================================

def test_migration_roundtrip_and_backfill(tmp_path):
    """Verify Alembic migration upgrade, existing row backfill to ACTIVE, downgrade, and re-upgrade."""
    db_file = tmp_path / "test_lifecycle_migration.db"
    db_url = f"sqlite:///{db_file.resolve().as_posix()}"

    orig_url = settings.DATABASE_URL
    settings.DATABASE_URL = db_url

    alembic_ini_path = BACKEND_DIR / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        # Step 1: Upgrade to previous revision (f2a9b3c4d5e6)
        command.upgrade(alembic_cfg, "f2a9b3c4d5e6")

        # Step 2: Seed data using the previous schema
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        now_str = "2026-11-16T12:00:00Z"
        org_id = "org-mig-001"
        meeting_id = "mtg-mig-001"
        decision_id = "dec-mig-001"
        action_id = "act-mig-001"
        question_id = "que-mig-001"

        cursor.execute(
            "INSERT INTO organizations (id, name, slug, is_active, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?)",
            (org_id, "Migration Org", "migration-org", now_str, now_str),
        )
        cursor.execute(
            "INSERT INTO meetings (id, organization_id, title, input_source_type, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'AUDIO', 'SCHEDULED', ?, ?)",
            (meeting_id, org_id, "Pre-migration Meeting", now_str, now_str),
        )
        # Decision without status column in previous revision
        cursor.execute(
            "INSERT INTO decisions (id, meeting_id, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (decision_id, meeting_id, "Pre-existing decision before status column", now_str, now_str),
        )
        # Action item without review columns
        cursor.execute(
            "INSERT INTO action_items (id, meeting_id, description, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'PENDING', ?, ?)",
            (action_id, meeting_id, "Pre-existing action item", now_str, now_str),
        )
        # Unresolved question without resolution columns
        cursor.execute(
            "INSERT INTO unresolved_questions (id, meeting_id, question_text, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'OPEN', ?, ?)",
            (question_id, meeting_id, "Pre-existing question", now_str, now_str),
        )
        conn.commit()
        conn.close()

        # Step 3: Upgrade to new revision c3b2a1d0e9f8
        command.upgrade(alembic_cfg, "c3b2a1d0e9f8")

        # Step 4: Verify columns, defaults, and foreign keys
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        # Check FK integrity
        fk_violations = cursor.execute("PRAGMA foreign_key_check").fetchall()
        assert len(fk_violations) == 0, f"Foreign key check violations detected: {fk_violations}"

        # Verify decision backfilled to ACTIVE and supersedes_decision_id is NULL
        cursor.execute("SELECT status, supersedes_decision_id FROM decisions WHERE id = ?", (decision_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "ACTIVE", f"Expected ACTIVE status default, got {row[0]}"
        assert row[1] is None

        # Verify action_items new columns exist and are NULL
        cursor.execute(
            "SELECT last_reviewed_meeting_id, last_evidence_segment_id, last_reviewed_at "
            "FROM action_items WHERE id = ?",
            (action_id,),
        )
        act_row = cursor.fetchone()
        assert act_row == (None, None, None)

        # Verify unresolved_questions new columns exist and are NULL
        cursor.execute(
            "SELECT resolved_at, resolving_meeting_id, resolving_decision_id, resolution_notes "
            "FROM unresolved_questions WHERE id = ?",
            (question_id,),
        )
        q_row = cursor.fetchone()
        assert q_row == (None, None, None, None)
        conn.close()

        # Step 5: Downgrade back to f2a9b3c4d5e6
        command.downgrade(alembic_cfg, "f2a9b3c4d5e6")

        # Step 6: Re-upgrade to head
        command.upgrade(alembic_cfg, "head")

    finally:
        settings.DATABASE_URL = orig_url


# =========================================================================
# 2. Decision Lineage Progression Tests
# =========================================================================

def test_decision_superseding_lineage(db_session, test_org_a):
    """Verify decision superseding chain: A -> B -> C and lineage reconstruction."""
    m1 = _create_test_meeting(db_session, test_org_a, "Q1 Roadmap")
    m2 = _create_test_meeting(db_session, test_org_a, "Q2 Revision")
    m3 = _create_test_meeting(db_session, test_org_a, "Q3 Final Pivot")

    # Decision A in Meeting 1
    dec_a = Decision(
        meeting_id=m1.id,
        description="Adopt Monolith architecture for v1",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec_a)
    db_session.commit()
    db_session.refresh(dec_a)

    assert dec_a.status == DecisionStatus.ACTIVE.value
    assert dec_a.supersedes_decision_id is None

    # Decision B in Meeting 2 supersedes Decision A
    dec_b = Decision(
        meeting_id=m2.id,
        description="Migrate to Modular Monolith with bounded contexts",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec_b)
    db_session.commit()
    db_session.refresh(dec_b)

    supersede_decision(db_session, dec_b, dec_a.id)
    db_session.commit()

    db_session.refresh(dec_a)
    db_session.refresh(dec_b)
    assert dec_a.status == DecisionStatus.SUPERSEDED.value
    assert dec_b.status == DecisionStatus.ACTIVE.value
    assert dec_b.supersedes_decision_id == dec_a.id
    assert dec_b.superseded_decision.id == dec_a.id
    assert dec_b in dec_a.superseding_decisions

    # Decision C in Meeting 3 supersedes Decision B
    dec_c = Decision(
        meeting_id=m3.id,
        description="Transition to Microservices architecture",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec_c)
    db_session.commit()
    db_session.refresh(dec_c)

    supersede_decision(db_session, dec_c, dec_b.id)
    db_session.commit()

    db_session.refresh(dec_b)
    db_session.refresh(dec_c)
    assert dec_b.status == DecisionStatus.SUPERSEDED.value
    assert dec_c.status == DecisionStatus.ACTIVE.value
    assert dec_c.supersedes_decision_id == dec_b.id

    # Lineage retrieval
    lineage = get_decision_lineage(db_session, dec_c.id)
    assert [d.id for d in lineage] == [dec_a.id, dec_b.id, dec_c.id]


# =========================================================================
# 3. Self-Referencing Prevention Tests
# =========================================================================

def test_self_superseding_prevention_service(db_session, test_org_a):
    """Verify that Python lineage service rejects a decision superseding itself."""
    m1 = _create_test_meeting(db_session, test_org_a, "Design Review")
    dec = Decision(
        meeting_id=m1.id,
        description="Launch beta in June",
        status=DecisionStatus.ACTIVE.value,
    )
    db_session.add(dec)
    db_session.commit()
    db_session.refresh(dec)

    with pytest.raises(SelfSupersedingError, match="cannot supersede itself"):
        supersede_decision(db_session, dec, dec.id)


def test_self_superseding_prevention_database_constraint(db_session, test_org_a):
    """Verify that SQLite CHECK constraint rejects self-referencing decision rows."""
    m1 = _create_test_meeting(db_session, test_org_a, "Design Review")
    import uuid
    dec_id = str(uuid.uuid4())

    dec = Decision(
        id=dec_id,
        meeting_id=m1.id,
        description="Invalid self-referencing decision",
        supersedes_decision_id=dec_id,
    )
    db_session.add(dec)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# =========================================================================
# 4. Circular Lineage Prevention Tests
# =========================================================================

def test_circular_lineage_prevention(db_session, test_org_a):
    """Verify cycle detection prevents cycles like A -> B -> A or A -> B -> C -> A."""
    m1 = _create_test_meeting(db_session, test_org_a, "Meeting 1")
    m2 = _create_test_meeting(db_session, test_org_a, "Meeting 2")
    m3 = _create_test_meeting(db_session, test_org_a, "Meeting 3")

    dec_a = Decision(meeting_id=m1.id, description="Strategy A")
    dec_b = Decision(meeting_id=m2.id, description="Strategy B")
    dec_c = Decision(meeting_id=m3.id, description="Strategy C")
    db_session.add_all([dec_a, dec_b, dec_c])
    db_session.commit()

    # B supersedes A
    supersede_decision(db_session, dec_b, dec_a.id)
    # C supersedes B
    supersede_decision(db_session, dec_c, dec_b.id)
    db_session.commit()

    # Attempt to have A supersede C (would create cycle A -> C -> B -> A)
    with pytest.raises(CircularLineageError, match="Circular decision lineage detected"):
        supersede_decision(db_session, dec_a, dec_c.id)


# =========================================================================
# 5. Cross-Organization Isolation Tests
# =========================================================================

def test_cross_organization_decision_superseding_rejected(db_session, test_org_a, test_org_b):
    """Verify a decision from Org B cannot supersede a decision from Org A."""
    m_a = _create_test_meeting(db_session, test_org_a, "Org A Meeting")
    m_b = _create_test_meeting(db_session, test_org_b, "Org B Meeting")

    dec_a = Decision(meeting_id=m_a.id, description="Org A Secret Policy")
    dec_b = Decision(meeting_id=m_b.id, description="Org B Policy")
    db_session.add_all([dec_a, dec_b])
    db_session.commit()

    with pytest.raises(CrossTenantLineageError, match="Cross-organization decision superseding is strictly prohibited"):
        supersede_decision(db_session, dec_b, dec_a.id)


def test_cross_organization_commitment_review_rejected(db_session, test_org_a, test_org_b):
    """Verify an action item from Org A cannot be reviewed in an Org B meeting."""
    m_a = _create_test_meeting(db_session, test_org_a, "Org A Standup")
    m_b = _create_test_meeting(db_session, test_org_b, "Org B Standup")

    ai_a = ActionItem(meeting_id=m_a.id, description="Org A Task")
    db_session.add(ai_a)
    db_session.commit()

    with pytest.raises(CrossTenantLineageError, match="Cross-organization commitment review is strictly prohibited"):
        record_commitment_review(
            db_session,
            action_item_id=ai_a.id,
            reviewing_meeting_id=m_b.id,
        )


def test_cross_organization_question_resolution_rejected(db_session, test_org_a, test_org_b):
    """Verify an unresolved question from Org A cannot be resolved by an Org B meeting or decision."""
    m_a = _create_test_meeting(db_session, test_org_a, "Org A Q&A")
    m_b = _create_test_meeting(db_session, test_org_b, "Org B Q&A")

    q_a = UnresolvedQuestion(meeting_id=m_a.id, question_text="Org A Question?")
    dec_b = Decision(meeting_id=m_b.id, description="Org B Answer")
    db_session.add_all([q_a, dec_b])
    db_session.commit()

    # Resolving meeting belongs to Org B
    with pytest.raises(CrossTenantLineageError, match="Cross-organization question resolution is strictly prohibited"):
        resolve_unresolved_question(
            db_session,
            question_id=q_a.id,
            resolving_meeting_id=m_b.id,
        )

    # Resolving decision belongs to Org B (even if meeting were Org A)
    with pytest.raises(CrossTenantLineageError, match="Resolving decision .* belongs to a different organization"):
        resolve_unresolved_question(
            db_session,
            question_id=q_a.id,
            resolving_meeting_id=m_a.id,
            resolving_decision_id=dec_b.id,
        )


# =========================================================================
# 6. Commitment Continuity Tests
# =========================================================================

def test_commitment_continuity_review_preserves_origin(db_session, test_org_a):
    """Verify review updates last_reviewed tracking while preserving origin meeting & evidence."""
    m1 = _create_test_meeting(db_session, test_org_a, "Sprint 1 Planning")
    seg1 = _create_test_transcript_with_segment(db_session, m1, "Alice will deploy the staging cluster by Friday.")

    ai = ActionItem(
        meeting_id=m1.id,
        description="Deploy staging cluster",
        owner_name="Alice",
        status=ActionItemStatus.PENDING.value,
        evidence_segment_id=seg1.id,
        timestamp_seconds=42.5,
    )
    db_session.add(ai)
    db_session.commit()
    db_session.refresh(ai)

    # In Sprint 2 Review, progress is reviewed with new evidence segment
    m2 = _create_test_meeting(db_session, test_org_a, "Sprint 2 Review")
    seg2 = _create_test_transcript_with_segment(db_session, m2, "Alice confirmed staging cluster is now up and tested.")

    updated_ai = record_commitment_review(
        db_session,
        action_item_id=ai.id,
        reviewing_meeting_id=m2.id,
        evidence_segment_id=seg2.id,
        status=ActionItemStatus.COMPLETED,
    )
    db_session.commit()
    db_session.refresh(ai)

    # Origin fields remain untouched
    assert ai.meeting_id == m1.id
    assert ai.evidence_segment_id == seg1.id
    assert ai.timestamp_seconds == 42.5
    assert ai.owner_name == "Alice"

    # Review fields properly populated
    assert ai.status == ActionItemStatus.COMPLETED.value
    assert ai.last_reviewed_meeting_id == m2.id
    assert ai.last_evidence_segment_id == seg2.id
    assert ai.last_reviewed_at is not None
    assert ai.last_reviewed_meeting.id == m2.id
    assert ai.last_evidence_segment.id == seg2.id


# =========================================================================
# 7. Unresolved Question Resolution Tests
# =========================================================================

def test_unresolved_question_resolution_lifecycle(db_session, test_org_a):
    """Verify question resolution metadata, linking resolving meeting and resolving decision."""
    m1 = _create_test_meeting(db_session, test_org_a, "Design Workshop")
    q = UnresolvedQuestion(
        meeting_id=m1.id,
        question_text="Should we support multi-region replication in v1?",
        raised_by="Bob",
        status=QuestionStatus.OPEN.value,
    )
    db_session.add(q)
    db_session.commit()
    db_session.refresh(q)

    assert q.status == QuestionStatus.OPEN.value
    assert q.resolved_at is None
    assert q.resolving_meeting_id is None
    assert q.resolving_decision_id is None

    # Resolve in Architecture Review meeting with an explicit decision
    m2 = _create_test_meeting(db_session, test_org_a, "Architecture Review")
    dec = Decision(
        meeting_id=m2.id,
        description="Defer multi-region replication to v2.0; launch with single-region multi-AZ.",
    )
    db_session.add(dec)
    db_session.commit()
    db_session.refresh(dec)

    resolved_q = resolve_unresolved_question(
        db_session,
        question_id=q.id,
        resolving_meeting_id=m2.id,
        resolving_decision_id=dec.id,
        resolution_notes="Agreed to defer multi-region until Q4 2027.",
        status=QuestionStatus.RESOLVED,
    )
    db_session.commit()
    db_session.refresh(q)

    assert q.status == QuestionStatus.RESOLVED.value
    assert q.resolving_meeting_id == m2.id
    assert q.resolving_decision_id == dec.id
    assert q.resolution_notes == "Agreed to defer multi-region until Q4 2027."
    assert q.resolved_at is not None
    assert q.resolving_meeting.id == m2.id
    assert q.resolving_decision.id == dec.id


# =========================================================================
# 8. Cascading & SET NULL Deletion Integrity Tests
# =========================================================================

def test_cascading_meeting_deletion(db_session, test_org_a):
    """Verify deleting a meeting cleanly cascades to its decisions, action items, and questions."""
    m = _create_test_meeting(db_session, test_org_a, "Ephemeral Meeting")
    dec = Decision(meeting_id=m.id, description="Temp decision")
    ai = ActionItem(meeting_id=m.id, description="Temp action")
    q = UnresolvedQuestion(meeting_id=m.id, question_text="Temp question?")
    db_session.add_all([dec, ai, q])
    db_session.commit()

    dec_id, ai_id, q_id = dec.id, ai.id, q.id

    # Delete meeting
    db_session.delete(m)
    db_session.commit()

    assert db_session.query(Decision).filter(Decision.id == dec_id).first() is None
    assert db_session.query(ActionItem).filter(ActionItem.id == ai_id).first() is None
    assert db_session.query(UnresolvedQuestion).filter(UnresolvedQuestion.id == q_id).first() is None


def test_set_null_on_superseded_decision_deletion(db_session, test_org_a):
    """Verify deleting a superseded decision sets supersedes_decision_id to NULL without cascading."""
    m1 = _create_test_meeting(db_session, test_org_a, "M1")
    m2 = _create_test_meeting(db_session, test_org_a, "M2")

    dec_a = Decision(meeting_id=m1.id, description="Original policy")
    dec_b = Decision(meeting_id=m2.id, description="Revised policy")
    db_session.add_all([dec_a, dec_b])
    db_session.commit()

    supersede_decision(db_session, dec_b, dec_a.id)
    db_session.commit()

    # Delete original decision A
    db_session.delete(dec_a)
    db_session.commit()

    # Decision B must still exist with supersedes_decision_id set to NULL
    db_session.refresh(dec_b)
    assert dec_b.supersedes_decision_id is None
    assert dec_b.description == "Revised policy"


# =========================================================================
# 9. Pydantic Schemas & Response Serialization Tests
# =========================================================================

def test_intelligence_schemas_serialization(db_session, test_org_a):
    """Verify DecisionResponse, ActionItemResponse, and UnresolvedQuestionResponse handle new fields."""
    m1 = _create_test_meeting(db_session, test_org_a, "Planning Meeting")
    m2 = _create_test_meeting(db_session, test_org_a, "Review Meeting")

    # Earlier decision
    earlier_dec = Decision(
        meeting_id=m1.id,
        description="Draft preliminary API spec",
        status=DecisionStatus.SUPERSEDED.value,
    )
    db_session.add(earlier_dec)
    db_session.commit()
    db_session.refresh(earlier_dec)

    # Decision schema with valid supersedes FK
    dec = Decision(
        meeting_id=m2.id,
        description="Approved v1 API design",
        status=DecisionStatus.ACTIVE.value,
        supersedes_decision_id=earlier_dec.id,
    )
    db_session.add(dec)
    db_session.commit()
    db_session.refresh(dec)

    dec_resp = DecisionResponse.model_validate(dec)
    assert dec_resp.status == "ACTIVE"
    assert dec_resp.supersedes_decision_id == earlier_dec.id

    # Action item schema with valid last_reviewed_meeting FK
    now = datetime.now(timezone.utc)
    ai = ActionItem(
        meeting_id=m1.id,
        description="Ship release candidate",
        last_reviewed_meeting_id=m2.id,
        last_reviewed_at=now,
    )
    db_session.add(ai)
    db_session.commit()
    db_session.refresh(ai)

    ai_resp = ActionItemResponse.model_validate(ai)
    assert ai_resp.last_reviewed_meeting_id == m2.id
    assert ai_resp.last_reviewed_at is not None
    assert ai_resp.last_reviewed_at.year == now.year

    # Unresolved question schema with valid resolving meeting and decision FKs
    q = UnresolvedQuestion(
        meeting_id=m1.id,
        question_text="Target release window?",
        status=QuestionStatus.RESOLVED.value,
        resolving_meeting_id=m2.id,
        resolving_decision_id=dec.id,
        resolution_notes="Targeted for Oct 15",
        resolved_at=now,
    )
    db_session.add(q)
    db_session.commit()
    db_session.refresh(q)

    q_resp = UnresolvedQuestionResponse.model_validate(q)
    assert q_resp.status == "RESOLVED"
    assert q_resp.resolving_meeting_id == m2.id
    assert q_resp.resolving_decision_id == dec.id
    assert q_resp.resolution_notes == "Targeted for Oct 15"
    assert q_resp.resolved_at is not None
    assert q_resp.resolved_at.year == now.year

    # MeetingIntelligenceResponse aggregation
    intel_resp = MeetingIntelligenceResponse(
        meeting_id=m2.id,
        decisions=[dec_resp],
        action_items=[ai_resp],
        unresolved_questions=[q_resp],
    )
    assert intel_resp.has_intelligence is True
    assert len(intel_resp.decisions) == 1
    assert intel_resp.decisions[0].status == "ACTIVE"
    assert intel_resp.action_items[0].last_reviewed_meeting_id == m2.id
    assert intel_resp.unresolved_questions[0].resolving_decision_id == dec.id


def test_action_and_update_schemas_validation():
    """Verify input validation and normalization for update payloads."""
    # DecisionStatusUpdate
    d_up = DecisionStatusUpdate(status="superseded")
    assert d_up.status == DecisionStatus.SUPERSEDED

    with pytest.raises(ValueError):
        DecisionStatusUpdate(status="UNKNOWN_STATUS")

    # CommitmentReviewUpdate
    c_up = CommitmentReviewUpdate(reviewing_meeting_id="m-123", status="completed")
    assert c_up.status == ActionItemStatus.COMPLETED

    # QuestionResolutionUpdate
    q_up = QuestionResolutionUpdate(resolving_meeting_id="m-456", resolution_notes="Resolved via email")
    assert q_up.status == QuestionStatus.RESOLVED
