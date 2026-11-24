import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.models.meeting import Meeting
from app.models.memory import AuditLog


def test_org_role_enum_contains_only_org_admin_and_member():
    """Verify that OrgRole contains organization roles (ORG_ADMIN, MEMBER, INDIVIDUAL) and does not contain SUPER_ADMIN."""
    role_names = set(OrgRole.__members__.keys())
    assert "ORG_ADMIN" in role_names
    assert "MEMBER" in role_names
    assert "INDIVIDUAL" in role_names
    assert hasattr(OrgRole, "SUPER_ADMIN") is False
    assert OrgRole.ORG_ADMIN.value == "ORG_ADMIN"
    assert OrgRole.MEMBER.value == "MEMBER"
    assert OrgRole.INDIVIDUAL.value == "INDIVIDUAL"


def test_user_is_super_admin_independent_global_privilege(db_session):
    """Verify that User.is_super_admin represents global system privilege independently of org roles."""
    org = Organization(name="Test Org", slug="test-org-privilege")
    db_session.add(org)
    db_session.commit()

    # User 1: Global super admin who is also ORG_ADMIN in this org
    global_admin = User(
        email="global.admin@example.com",
        full_name="Global Super Admin",
        is_super_admin=True,
    )
    # User 2: Standard user who is ORG_ADMIN in this org (NOT global super admin)
    tenant_admin = User(
        email="tenant.admin@example.com",
        full_name="Tenant Admin Only",
        is_super_admin=False,
    )
    # User 3: Standard user who is a regular MEMBER
    regular_member = User(
        email="member@example.com",
        full_name="Regular Member",
        is_super_admin=False,
    )
    db_session.add_all([global_admin, tenant_admin, regular_member])
    db_session.commit()

    # Assign memberships
    m_global = OrganizationMembership(
        organization_id=org.id,
        user_id=global_admin.id,
        role=OrgRole.ORG_ADMIN.value,
    )
    m_tenant = OrganizationMembership(
        organization_id=org.id,
        user_id=tenant_admin.id,
        role=OrgRole.ORG_ADMIN.value,
    )
    m_regular = OrganizationMembership(
        organization_id=org.id,
        user_id=regular_member.id,
        role=OrgRole.MEMBER.value,
    )
    db_session.add_all([m_global, m_tenant, m_regular])
    db_session.commit()

    # Verify distinctions
    assert global_admin.is_super_admin is True
    assert m_global.role == OrgRole.ORG_ADMIN.value

    assert tenant_admin.is_super_admin is False
    assert m_tenant.role == OrgRole.ORG_ADMIN.value

    assert regular_member.is_super_admin is False
    assert m_regular.role == OrgRole.MEMBER.value


def test_unique_membership_constraint_enforced(db_session):
    """Verify that a user cannot have duplicate memberships in the same organization."""
    org = Organization(name="Constraint Org", slug="constraint-org")
    user = User(email="test.dup@example.com", full_name="Dup Tester")
    db_session.add_all([org, user])
    db_session.commit()

    m1 = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.MEMBER.value,
    )
    db_session.add(m1)
    db_session.commit()

    m2 = OrganizationMembership(
        organization_id=org.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
    )
    db_session.add(m2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_development_membership_and_data_integrity():
    """Verify that development user, organization, meetings, audit logs, and membership remain intact."""
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        # 1. Dev user checks
        dev_user = db.execute(
            select(User).where(User.email == "dev@local.internal")
        ).scalar_one_or_none()
        assert dev_user is not None
        assert dev_user.full_name == "Local Developer"
        assert dev_user.is_super_admin is True
        assert dev_user.hashed_password is None  # Password remains unchanged

        # 2. Dev organization checks
        dev_org = db.execute(
            select(Organization).where(Organization.slug == "default-dev-org")
        ).scalar_one_or_none()
        assert dev_org is not None
        assert dev_org.name == "Default Development Organization"

        # 3. Seeded membership checks
        membership = db.execute(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == dev_org.id,
                OrganizationMembership.user_id == dev_user.id,
            )
        ).scalar_one_or_none()
        assert membership is not None
        assert membership.role == OrgRole.ORG_ADMIN.value
        assert membership.is_active is True

        # Exactly one membership exists for dev user
        all_dev_memberships = db.execute(
            select(OrganizationMembership).where(OrganizationMembership.user_id == dev_user.id)
        ).scalars().all()
        assert len(all_dev_memberships) == 1

        # 4. Meeting and audit record preservation
        meeting = db.execute(
            select(Meeting).where(Meeting.organization_id == dev_org.id)
        ).first()
        assert meeting is not None

        audit_log = db.execute(
            select(AuditLog).where(AuditLog.organization_id == dev_org.id)
        ).first()
        assert audit_log is not None

    finally:
        db.close()
