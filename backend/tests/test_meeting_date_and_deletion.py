"""Tests for Meeting Date/Time, Search/Filter/Pagination, and Safe Full Meeting Deletion."""
import os
import shutil
from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting, MeetingParticipant, MediaAsset, InputSourceType, MeetingStatus
from app.models.project import Project
from app.models.memory import AuditLog, TranscriptChunkRecord
from app.core.security import hash_password, create_access_token
from app.services.media_storage import get_meeting_storage_dir


@pytest.fixture
def meeting_test_setup(db_session: Session):
    org = Organization(name="Audit Alpha", slug="audit-alpha", is_active=True)
    db_session.add(org)
    db_session.commit()

    admin_user = User(
        email="admin_audit@alpha.com",
        full_name="Admin Audit",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    member_user = User(
        email="member_audit@alpha.com",
        full_name="Member Audit",
        hashed_password=hash_password("Password123!"),
        is_active=True,
    )
    db_session.add_all([admin_user, member_user])
    db_session.commit()

    mem_admin = OrganizationMembership(
        organization_id=org.id,
        user_id=admin_user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    mem_member = OrganizationMembership(
        organization_id=org.id,
        user_id=member_user.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add_all([mem_admin, mem_member])
    db_session.commit()

    token_admin = create_access_token(subject=admin_user.id, token_version=admin_user.token_version)
    token_member = create_access_token(subject=member_user.id, token_version=member_user.token_version)

    headers_admin = {
        "Authorization": f"Bearer {token_admin}",
        "X-Organization-Id": org.id,
    }
    headers_member = {
        "Authorization": f"Bearer {token_member}",
        "X-Organization-Id": org.id,
    }

    return {
        "org": org,
        "admin": admin_user,
        "member": member_user,
        "headers_admin": headers_admin,
        "headers_member": headers_member,
    }


def test_meeting_date_filtering_and_pagination(client: TestClient, meeting_test_setup, db_session: Session):
    headers = meeting_test_setup["headers_admin"]
    org = meeting_test_setup["org"]

    proj = Project(
        organization_id=org.id,
        name="Security Audit",
        slug="security-audit",
    )
    db_session.add(proj)
    db_session.commit()

    base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    meetings = []
    for i in range(15):
        m = Meeting(
            organization_id=org.id,
            title=f"Sprint Review Session {i+1}",
            description=f"Detailed review of sprint milestone {i+1}",
            actual_start_time=base_time + timedelta(days=i),
            category="Sprint Planning" if i % 2 == 0 else "Architecture Review",
            project_id=proj.id if i < 10 else None,
            status=MeetingStatus.COMPLETED.value,
            input_source_type=InputSourceType.AUDIO.value,
        )
        meetings.append(m)
    db_session.add_all(meetings)
    db_session.commit()

    # 1. Test Search by Title
    res_search = client.get("/api/v1/media/meetings?search=Session 5", headers=headers)
    assert res_search.status_code == 200
    items = res_search.json()
    assert len(items) == 1
    assert "Session 5" in items[0]["title"]

    # 2. Test Filter by Category
    res_cat = client.get("/api/v1/media/meetings?category=Sprint Planning", headers=headers)
    assert res_cat.status_code == 200
    cat_items = res_cat.json()
    assert len(cat_items) == 8
    assert all(it["category"] == "Sprint Planning" for it in cat_items)

    # 3. Test Filter by Project
    res_proj = client.get(f"/api/v1/media/meetings?project_id={proj.id}", headers=headers)
    assert res_proj.status_code == 200
    assert len(res_proj.json()) == 10

    # 4. Test Pagination
    res_paginated = client.get("/api/v1/media/meetings?page=1&limit=5&paginate=true", headers=headers)
    assert res_paginated.status_code == 200
    pdata = res_paginated.json()
    assert pdata["total"] == 15
    assert pdata["page"] == 1
    assert pdata["pages"] == 3
    assert len(pdata["items"]) == 5


def test_safe_full_meeting_deletion(client: TestClient, meeting_test_setup, db_session: Session):
    headers_admin = meeting_test_setup["headers_admin"]
    headers_member = meeting_test_setup["headers_member"]
    org = meeting_test_setup["org"]

    # Create meeting
    m = Meeting(
        organization_id=org.id,
        title="Confidential Mergers Call",
        category="Other",
        status=MeetingStatus.COMPLETED.value,
        input_source_type=InputSourceType.AUDIO.value,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(m)

    # Create a mock physical storage directory
    meeting_dir = get_meeting_storage_dir(m.id)
    os.makedirs(meeting_dir, exist_ok=True)
    mock_file = meeting_dir / "raw_recording.wav"
    mock_file.write_text("dummy audio data")
    assert mock_file.exists()

    # Create participant and media asset
    part = MeetingParticipant(meeting_id=m.id, display_name="Eve Auditor")
    asset = MediaAsset(
        meeting_id=m.id,
        file_name="raw_recording.wav",
        storage_path=str(mock_file),
        media_type="AUDIO",
        mime_type="audio/wav",
        is_raw_deleted=False
    )
    db_session.add_all([part, asset])
    db_session.commit()

    # 1. Non-admin (MEMBER) cannot delete meeting
    res_denied = client.delete(f"/api/v1/media/meetings/{m.id}", headers=headers_member)
    assert res_denied.status_code == 403

    # 2. ORG_ADMIN deletes meeting
    res_del = client.delete(f"/api/v1/media/meetings/{m.id}", headers=headers_admin)
    assert res_del.status_code == 200
    assert res_del.json()["meeting_id"] == m.id

    # 3. Verify meeting is gone
    res_get = client.get(f"/api/v1/media/meetings/{m.id}", headers=headers_admin)
    assert res_get.status_code == 404

    # 4. Verify physical storage folder removed
    assert not meeting_dir.exists()

    # 5. Verify audit log entry
    audit = db_session.query(AuditLog).filter(
        AuditLog.organization_id == org.id,
        AuditLog.action == "MEETING_DELETED",
        AuditLog.entity_id == m.id
    ).first()
    assert audit is not None
    assert "Confidential Mergers Call" in audit.details
