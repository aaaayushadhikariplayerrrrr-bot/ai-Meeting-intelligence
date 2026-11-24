"""Tests for Project-Scoped Ask AI and Organizational Memory Retrieval."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, InputSourceType, MeetingStatus
from app.models.project import Project
from app.models.intelligence import Decision
from app.core.security import hash_password, create_access_token


@pytest.fixture
def rag_project_setup(db_session: Session):
    org = Organization(name="RAG Org", slug="rag-org", is_active=True)
    org_other = Organization(name="Other Org", slug="other-org", is_active=True)
    db_session.add_all([org, org_other])
    db_session.commit()

    user = User(
        email="ai_researcher@org.com",
        full_name="AI Researcher",
        hashed_password=hash_password("Password123!"),
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

    token = create_access_token(subject=user.id, token_version=user.token_version)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Organization-Id": org.id,
    }

    # Create two projects in Org
    proj_a = Project(organization_id=org.id, name="Cloud Migration", slug="cloud-migration")
    proj_b = Project(organization_id=org.id, name="Mobile App", slug="mobile-app")
    proj_foreign = Project(organization_id=org_other.id, name="Foreign Project", slug="foreign-project")
    db_session.add_all([proj_a, proj_b, proj_foreign])
    db_session.commit()

    # Create meeting for Project A with a decision
    meeting_a = Meeting(
        organization_id=org.id,
        project_id=proj_a.id,
        title="Cloud Architecture Sync",
        status=MeetingStatus.COMPLETED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    # Create meeting for Project B with a decision
    meeting_b = Meeting(
        organization_id=org.id,
        project_id=proj_b.id,
        title="Mobile Navigation Review",
        status=MeetingStatus.COMPLETED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db_session.add_all([meeting_a, meeting_b])
    db_session.commit()

    dec_a = Decision(
        meeting_id=meeting_a.id,
        description="Adopt Kubernetes for all container workloads in production.",
        status="ACTIVE",
    )
    dec_b = Decision(
        meeting_id=meeting_b.id,
        description="Use React Native for cross-platform mobile client development.",
        status="ACTIVE",
    )
    db_session.add_all([dec_a, dec_b])
    db_session.commit()

    return {
        "org": org,
        "headers": headers,
        "proj_a": proj_a,
        "proj_b": proj_b,
        "proj_foreign": proj_foreign,
    }


def test_project_scoped_ask_ai_and_search(client: TestClient, rag_project_setup):
    headers = rag_project_setup["headers"]
    proj_a = rag_project_setup["proj_a"]
    proj_b = rag_project_setup["proj_b"]
    proj_foreign = rag_project_setup["proj_foreign"]

    # 1. Querying with foreign project ID returns 404
    res_foreign = client.post(
        "/api/v1/memory/query",
        headers=headers,
        json={"question": "What architecture decisions were made?", "project_id": proj_foreign.id}
    )
    assert res_foreign.status_code == 404

    # 2. Querying with Project A scopes decisions to Project A only
    res_a = client.post(
        "/api/v1/memory/query",
        headers=headers,
        json={"question": "What architecture decisions were made?", "project_id": proj_a.id}
    )
    assert res_a.status_code == 200
    data_a = res_a.json()
    # Decisions in Project A must only contain Kubernetes, never React Native
    dec_texts_a = [d["description"] for d in data_a.get("structured_decisions", [])]
    if dec_texts_a:
        assert any("Kubernetes" in t for t in dec_texts_a)
        assert not any("React Native" in t for t in dec_texts_a)

    # 3. Querying with Project B scopes decisions to Project B only
    res_b = client.post(
        "/api/v1/memory/query",
        headers=headers,
        json={"question": "What decisions were made regarding mobile?", "project_id": proj_b.id}
    )
    assert res_b.status_code == 200
    data_b = res_b.json()
    dec_texts_b = [d["description"] for d in data_b.get("structured_decisions", [])]
    if dec_texts_b:
        assert any("React Native" in t for t in dec_texts_b)
        assert not any("Kubernetes" in t for t in dec_texts_b)
