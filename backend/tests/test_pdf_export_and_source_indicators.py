"""Tests for PDF Meeting Report Export and Ask AI Answer Source Indicators."""

import io
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingParticipant, InputSourceType, MeetingStatus
from app.models.project import Project
from app.models.intelligence import (
    MeetingSummary,
    Topic,
    Decision,
    ActionItem,
    Risk,
    UnresolvedQuestion,
    DecisionStatus,
    ActionItemStatus,
)
from app.models.transcription import Transcript, TranscriptSegment, TranscriptType
from app.core.security import hash_password, create_access_token
from app.services.report_generator import generate_meeting_pdf_report
from app.schemas.memory import (
    RAGQueryResponse,
    ExternalKnowledgeResult,
    ExternalKnowledgeCitation,
    AbstentionStatus,
    RetrievalDiagnostics,
    Citation,
)


@pytest.fixture
def pdf_test_setup(db_session: Session):
    org = Organization(name="Report Org", slug="report-org", is_active=True)
    other_org = Organization(name="Other Org", slug="other-org", is_active=True)
    db_session.add_all([org, other_org])
    db_session.commit()

    user = User(
        email="reporter@example.com",
        full_name="Reporter User",
        hashed_password=hash_password("Pass123!"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    mem = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(mem)
    db_session.commit()

    project = Project(
        organization_id=org.id,
        name="Security Infrastructure",
        slug="security-infra",
        description="Project for security upgrades",
    )
    db_session.add(project)
    db_session.commit()

    meeting = Meeting(
        organization_id=org.id,
        project_id=project.id,
        title="Infrastructure Security Review",
        description="Annual review of security architecture and IAM roles.",
        input_source_type=InputSourceType.AUDIO.value,
        status=MeetingStatus.COMPLETED.value,
        actual_start_time=datetime(2026, 11, 20, 14, 0, tzinfo=timezone.utc),
        category="SECURITY",
    )
    db_session.add(meeting)
    db_session.commit()

    participant = MeetingParticipant(
        meeting_id=meeting.id,
        display_name="Security Lead",
        email="sec@example.com",
        role="PRESENTER",
    )
    db_session.add(participant)

    summary = MeetingSummary(
        meeting_id=meeting.id,
        executive_summary="Team reviewed MFA rollout and agreed on mandatory hardware tokens.",
        key_takeaways='["Roll out FIDO2 keys", "Deprecate SMS MFA"]',
    )
    db_session.add(summary)

    decision = Decision(
        meeting_id=meeting.id,
        description="Enforce FIDO2 security keys for all production engineers",
        status=DecisionStatus.ACTIVE.value,
        rationale_or_context="Prevents phishing and credential harvesting attacks.",
        timestamp_seconds=120.0,
    )
    db_session.add(decision)

    action = ActionItem(
        meeting_id=meeting.id,
        description="Distribute hardware tokens to cloud team",
        owner_name="Security Lead",
        status=ActionItemStatus.PENDING.value,
        timestamp_seconds=180.0,
    )
    db_session.add(action)

    question = UnresolvedQuestion(
        meeting_id=meeting.id,
        question_text="Will contractors receive YubiKeys or virtual authenticators?",
        raised_by="Alex",
    )
    db_session.add(question)

    risk = Risk(
        meeting_id=meeting.id,
        description="Key provisioning delays could impact Q1 deployment",
        severity="MEDIUM",
    )
    db_session.add(risk)

    transcript = Transcript(
        meeting_id=meeting.id,
        transcript_type=TranscriptType.CLEANED_REDACTED.value,
        version=1,
    )
    db_session.add(transcript)
    db_session.commit()

    segment = TranscriptSegment(
        transcript_id=transcript.id,
        sequence_order=0,
        start_time=120.0,
        end_time=145.0,
        text="We are officially mandating FIDO2 security keys across all production access points.",
        language_code="en",
    )
    db_session.add(segment)
    db_session.commit()

    token = create_access_token(subject=user.id, token_version=user.token_version)

    return {
        "org": org,
        "other_org": other_org,
        "user": user,
        "token": token,
        "meeting": meeting,
        "project": project,
    }


def test_pdf_report_generator_unit():
    """Unit test verify PDF generation bytes and header."""
    meeting_data = {
        "title": "Strategy Sync",
        "actual_start_time": "2026-11-20T10:00:00Z",
        "project_name": "Core Platform",
        "category": "STRATEGY",
        "status": "PROCESSED",
    }
    intelligence_data = {
        "summary": {"summary_text": "Executive summary paragraph."},
        "topics": [{"title": "Roadmap", "description": "Reviewing milestones", "start_time": 0.0, "end_time": 60.0}],
        "decisions": [{"decision_text": "Adopt v2 API", "status": "ACTIVE"}],
        "action_items": [{"description": "Write specs", "owner_name": "Dev", "status": "PENDING"}],
    }
    participants = [{"display_name": "Alice", "email": "alice@test.com", "role": "ATTENDEE", "is_speaker": False}]
    evidence = [{"start_time": 10.0, "end_time": 25.0, "speaker": "Alice", "text": "We will ship v2."}]

    pdf_bytes = generate_meeting_pdf_report(meeting_data, intelligence_data, participants, evidence)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


def test_pdf_report_export_api_endpoint(client: TestClient, pdf_test_setup):
    """Integration test for GET /api/v1/media/meetings/{id}/export/pdf."""
    setup = pdf_test_setup
    meeting_id = setup["meeting"].id
    headers = {"Authorization": f"Bearer {setup['token']}"}

    response = client.get(f"/api/v1/media/meetings/{meeting_id}/export/pdf", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers.get("content-disposition", "")
    assert response.content.startswith(b"%PDF")
    assert len(response.content) > 1500


def test_pdf_report_export_direct_meetings_route(client: TestClient, pdf_test_setup):
    """Integration test for GET /api/v1/meetings/{id}/export/pdf."""
    setup = pdf_test_setup
    meeting_id = setup["meeting"].id
    headers = {"Authorization": f"Bearer {setup['token']}"}

    response = client.get(f"/api/v1/meetings/{meeting_id}/export/pdf", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")


def test_pdf_report_export_tenant_isolation(client: TestClient, db_session: Session, pdf_test_setup):
    """Verify other organization cannot export PDF for this meeting."""
    setup = pdf_test_setup
    other_org = setup["other_org"]

    other_user = User(
        email="intruder@other.com",
        full_name="Intruder User",
        hashed_password=hash_password("Pass123!"),
        is_active=True,
    )
    db_session.add(other_user)
    db_session.commit()

    other_mem = OrganizationMembership(
        organization_id=other_org.id,
        user_id=other_user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(other_mem)
    db_session.commit()

    other_token = create_access_token(subject=other_user.id, token_version=other_user.token_version)
    headers = {"Authorization": f"Bearer {other_token}"}

    response = client.get(f"/api/v1/media/meetings/{setup['meeting'].id}/export/pdf", headers=headers)
    assert response.status_code == 404


def test_answer_source_type_classifications():
    """Verify that RAGQueryResponse source classifications conform strictly to requirements."""
    diagnostics = RetrievalDiagnostics(
        query="Test query",
        total_candidates_retrieved=1,
        evidence_items_count=1,
    )
    citation = Citation(
        citation_id="cit-1",
        chunk_id="chunk-1",
        meeting_id="meet-1",
        meeting_title="Meet 1",
        transcript_id="tr-1",
        start_time=10.0,
        end_time=20.0,
        quoted_text="We decided to use Redis.",
    )
    ext_knowledge = ExternalKnowledgeResult(
        subject="103rd Constitutional Amendment",
        summary="Introduced 10% EWS reservation in India.",
        citations=[ExternalKnowledgeCitation(source_title="Ministry of Law", source_url="https://law.gov.in")],
        is_available=True,
    )

    # 1. Meeting Evidence Only
    resp_meeting = RAGQueryResponse(
        question="What database was chosen?",
        answer="Redis was chosen for caching.",
        citations=[citation],
        diagnostics=diagnostics,
        answer_source_type="MEETING_EVIDENCE",
    )
    assert resp_meeting.answer_source_type == "MEETING_EVIDENCE"

    # 2. External Knowledge Only
    resp_ext = RAGQueryResponse(
        question="What is the 103rd Amendment?",
        answer="The 103rd Amendment introduced EWS reservation.",
        citations=[],
        external_knowledge=ext_knowledge,
        diagnostics=diagnostics,
        answer_source_type="EXTERNAL_KNOWLEDGE",
    )
    assert resp_ext.answer_source_type == "EXTERNAL_KNOWLEDGE"

    # 3. Combined Answer
    resp_combined = RAGQueryResponse(
        question="Explain the 103rd Amendment mentioned in the meeting",
        answer="The meeting mentioned the 103rd Amendment; it introduced EWS reservation.",
        citations=[citation],
        external_knowledge=ext_knowledge,
        diagnostics=diagnostics,
        answer_source_type="COMBINED",
    )
    assert resp_combined.answer_source_type == "COMBINED"

    # 4. Insufficient Evidence
    resp_insufficient = RAGQueryResponse(
        question="What was said about quantum computing?",
        answer="I do not have sufficient evidence.",
        citations=[],
        is_abstained=True,
        abstention_status=AbstentionStatus.NO_EVIDENCE,
        diagnostics=diagnostics,
        answer_source_type="INSUFFICIENT_EVIDENCE",
    )
    assert resp_insufficient.answer_source_type == "INSUFFICIENT_EVIDENCE"
