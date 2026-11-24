"""Tests for Projects, Meeting Categories, and Participant Management."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingParticipant, InputSourceType, MeetingStatus
from app.models.project import Project
from app.core.security import hash_password, create_access_token


@pytest.fixture
def test_setup(db_session: Session):
    # Create Organization A and B
    org_a = Organization(name="Org Alpha", slug="org-alpha", is_active=True)
    org_b = Organization(name="Org Beta", slug="org-beta", is_active=True)
    db_session.add_all([org_a, org_b])
    db_session.commit()

    # User A (Admin of Org A)
    user_a = User(
        email="admin_a@alpha.com",
        full_name="Admin Alpha",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    # User B (Admin of Org B)
    user_b = User(
        email="admin_b@beta.com",
        full_name="Admin Beta",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    db_session.add_all([user_a, user_b])
    db_session.commit()

    mem_a = OrganizationMembership(
        organization_id=org_a.id,
        user_id=user_a.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    mem_b = OrganizationMembership(
        organization_id=org_b.id,
        user_id=user_b.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add_all([mem_a, mem_b])
    db_session.commit()

    token_a = create_access_token(subject=user_a.id, token_version=user_a.token_version)
    token_b = create_access_token(subject=user_b.id, token_version=user_b.token_version)

    headers_a = {
        "Authorization": f"Bearer {token_a}",
        "X-Organization-Id": org_a.id,
    }
    headers_b = {
        "Authorization": f"Bearer {token_b}",
        "X-Organization-Id": org_b.id,
    }

    return {
        "org_a": org_a,
        "org_b": org_b,
        "user_a": user_a,
        "user_b": user_b,
        "headers_a": headers_a,
        "headers_b": headers_b,
    }


def test_project_crud_and_tenant_isolation(client: TestClient, test_setup):
    headers_a = test_setup["headers_a"]
    headers_b = test_setup["headers_b"]

    # 1. Create Project in Org A
    res = client.post(
        "/api/v1/projects",
        headers=headers_a,
        json={"name": "Core Platform Rewrite", "description": "Migrating to modern stack"}
    )
    assert res.status_code == 201
    proj_a = res.json()
    assert proj_a["name"] == "Core Platform Rewrite"
    assert proj_a["slug"] == "core-platform-rewrite"
    proj_id = proj_a["id"]

    # 2. Org B should not see Org A's project
    res_b_list = client.get("/api/v1/projects", headers=headers_b)
    assert res_b_list.status_code == 200
    assert len(res_b_list.json()) == 0

    res_b_get = client.get(f"/api/v1/projects/{proj_id}", headers=headers_b)
    assert res_b_get.status_code == 404

    # 3. Update project in Org A
    res_update = client.patch(
        f"/api/v1/projects/{proj_id}",
        headers=headers_a,
        json={"name": "Core Platform 2.0"}
    )
    assert res_update.status_code == 200
    assert res_update.json()["name"] == "Core Platform 2.0"

    # 4. Delete project
    res_del = client.delete(f"/api/v1/projects/{proj_id}", headers=headers_a)
    assert res_del.status_code == 200

    # 5. Verify deleted
    res_after = client.get(f"/api/v1/projects/{proj_id}", headers=headers_a)
    assert res_after.status_code == 404


def test_meeting_category_and_participants(client: TestClient, test_setup, db_session: Session):
    headers_a = test_setup["headers_a"]
    org_a = test_setup["org_a"]

    # Create a meeting in Org A
    meeting = Meeting(
        organization_id=org_a.id,
        title="Q4 Sprint Planning",
        category="Sprint Planning",
        status=MeetingStatus.SCHEDULED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db_session.add(meeting)
    db_session.commit()
    db_session.refresh(meeting)

    # Add participants
    res_add_p1 = client.post(
        f"/api/v1/media/meetings/{meeting.id}/participants",
        headers=headers_a,
        json={"display_name": "Alice Engineer", "email": "alice@company.com", "role": "Lead Architect"}
    )
    assert res_add_p1.status_code == 201
    p1 = res_add_p1.json()
    assert p1["display_name"] == "Alice Engineer"

    res_add_p2 = client.post(
        f"/api/v1/media/meetings/{meeting.id}/participants",
        headers=headers_a,
        json={"display_name": "Bob Product", "email": "bob@company.com", "role": "PM"}
    )
    assert res_add_p2.status_code == 201
    p2 = res_add_p2.json()

    # List participants
    res_list = client.get(f"/api/v1/media/meetings/{meeting.id}/participants", headers=headers_a)
    assert res_list.status_code == 200
    parts = res_list.json()
    assert len(parts) == 2

    # Remove participant
    res_del_p = client.delete(
        f"/api/v1/media/meetings/{meeting.id}/participants/{p1['id']}",
        headers=headers_a
    )
    assert res_del_p.status_code == 200

    # Verify only 1 remaining
    res_list_after = client.get(f"/api/v1/media/meetings/{meeting.id}/participants", headers=headers_a)
    assert len(res_list_after.json()) == 1
    assert res_list_after.json()[0]["id"] == p2["id"]
