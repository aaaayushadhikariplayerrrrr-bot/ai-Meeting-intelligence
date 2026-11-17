import sqlite3
from typing import Dict, Optional
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.database.base import Base
from app.database.session import get_db
from app.main import app as fastapi_app
from app.models.identity import User, Organization, OrganizationMembership, OrgRole
from app.core.security import create_access_token, hash_password
import app.models  # noqa: F401

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(test_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Ensure SQLite foreign key constraints are enforced in tests."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine
)


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state and restore clock before and after each test for test isolation."""
    from app.core.rate_limit import login_rate_limiter, register_rate_limiter
    login_rate_limiter.reset()
    register_rate_limiter.reset()
    yield
    login_rate_limiter.reset()
    register_rate_limiter.reset()


@pytest.fixture(scope="function")
def db_session():
    """Provide a fresh database session with tables created per test function."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    fastapi_app.dependency_overrides[get_db] = override_get_db
    with TestClient(fastapi_app) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()


def make_auth_headers(user: User, org: Optional[Organization] = None) -> Dict[str, str]:
    """Generate authentication headers with valid JWT Bearer token and optional org header."""
    token = create_access_token(subject=user.id, token_version=getattr(user, "token_version", 1))
    headers = {"Authorization": f"Bearer {token}"}
    if org:
        headers["X-Organization-Id"] = org.id
    return headers


@pytest.fixture(scope="function")
def test_org_a(db_session) -> Organization:
    """Fixture providing primary test Organization A."""
    org = Organization(name="Test Org A", slug="test-org-a", is_active=True)
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture(scope="function")
def test_org_b(db_session) -> Organization:
    """Fixture providing secondary test Organization B."""
    org = Organization(name="Test Org B", slug="test-org-b", is_active=True)
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture(scope="function")
def org_admin_user(db_session, test_org_a) -> User:
    """Fixture providing an active ORG_ADMIN user in test_org_a."""
    user = User(
        email="admin_a@example.com",
        full_name="Admin Alpha",
        hashed_password=hash_password("AdminPass123!"),
        is_super_admin=False,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=test_org_a.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def regular_member_user(db_session, test_org_a) -> User:
    """Fixture providing an active MEMBER user in test_org_a."""
    user = User(
        email="member_a@example.com",
        full_name="Member Alpha",
        hashed_password=hash_password("MemberPass123!"),
        is_super_admin=False,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=test_org_a.id,
        user_id=user.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def multi_org_user(db_session, test_org_a, test_org_b) -> User:
    """Fixture providing a user belonging to both Org A and Org B."""
    user = User(
        email="multiorg@example.com",
        full_name="Multi Org User",
        hashed_password=hash_password("MultiPass123!"),
        is_super_admin=False,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m1 = OrganizationMembership(
        organization_id=test_org_a.id,
        user_id=user.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    m2 = OrganizationMembership(
        organization_id=test_org_b.id,
        user_id=user.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add_all([m1, m2])
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def super_admin_no_membership(db_session) -> User:
    """Fixture providing a global Super Admin without any organization membership."""
    user = User(
        email="super_global@example.com",
        full_name="Global Super Admin",
        hashed_password=hash_password("SuperPass123!"),
        is_super_admin=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def super_admin_member(db_session, test_org_a) -> User:
    """Fixture providing a global Super Admin with MEMBER role in test_org_a."""
    user = User(
        email="super_member@example.com",
        full_name="Super Admin Member",
        hashed_password=hash_password("SuperPass123!"),
        is_super_admin=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=test_org_a.id,
        user_id=user.id,
        role=OrgRole.MEMBER.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def super_admin_org_admin(db_session, test_org_a) -> User:
    """Fixture providing a global Super Admin with ORG_ADMIN role in test_org_a."""
    user = User(
        email="super_admin_org@example.com",
        full_name="Super Admin Org Admin",
        hashed_password=hash_password("SuperPass123!"),
        is_super_admin=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()

    m = OrganizationMembership(
        organization_id=test_org_a.id,
        user_id=user.id,
        role=OrgRole.ORG_ADMIN.value,
        is_active=True,
    )
    db_session.add(m)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def org_admin_headers(org_admin_user, test_org_a) -> Dict[str, str]:
    """Headers for authenticated ORG_ADMIN in test_org_a."""
    return make_auth_headers(org_admin_user, test_org_a)


@pytest.fixture(scope="function")
def regular_member_headers(regular_member_user, test_org_a) -> Dict[str, str]:
    """Headers for authenticated MEMBER in test_org_a."""
    return make_auth_headers(regular_member_user, test_org_a)
