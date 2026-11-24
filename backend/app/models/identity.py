import enum
from sqlalchemy import Column, String, Boolean, Text, ForeignKey, UniqueConstraint, Integer
from sqlalchemy.orm import relationship
from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin


class OrgRole(str, enum.Enum):
    """Organization-level membership roles."""
    ORG_ADMIN = "ORG_ADMIN"
    ADMIN = "ORG_ADMIN"  # Canonical alias for ORG_ADMIN
    MEMBER = "MEMBER"
    INDIVIDUAL = "INDIVIDUAL"


class User(UUIDMixin, TimestampMixin, Base):
    """User account entity.

    Global system administration (is_super_admin) is distinguished from
    organization-level roles which are captured in OrganizationMembership.
    """
    __tablename__ = "users"

    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Placeholder for auth phase
    is_super_admin = Column(Boolean, default=False, nullable=False)  # Global system privilege
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False, server_default="1")
    verification_token = Column(String(255), nullable=True)
    token_version = Column(Integer, default=1, nullable=False, server_default="1")

    # Relationships
    memberships = relationship(
        "OrganizationMembership",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    created_meetings = relationship(
        "Meeting",
        back_populates="created_by_user",
        foreign_keys="Meeting.created_by_user_id"
    )
    action_items = relationship(
        "ActionItem",
        back_populates="owner_user"
    )


class Organization(UUIDMixin, TimestampMixin, Base):
    """Organization / workspace tenant for organizational data isolation."""
    __tablename__ = "organizations"

    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    company_domain = Column(String(255), nullable=True, index=True)
    is_domain_verified = Column(Boolean, default=True, nullable=False, server_default="1")
    domain_verification_token = Column(String(255), nullable=True)
    domain_verification_method = Column(String(50), nullable=True)
    account_type = Column(String(50), default="ORGANIZATION", nullable=False, server_default="ORGANIZATION")

    # Relationships
    memberships = relationship(
        "OrganizationMembership",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
    meetings = relationship(
        "Meeting",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
    knowledge_records = relationship(
        "KnowledgeRecord",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
    audit_logs = relationship(
        "AuditLog",
        back_populates="organization",
        cascade="all, delete-orphan"
    )
    projects = relationship(
        "Project",
        back_populates="organization",
        cascade="all, delete-orphan"
    )


class OrganizationMembership(UUIDMixin, TimestampMixin, Base):
    """Mapping between User and Organization defining tenant-level role."""
    __tablename__ = "organization_memberships"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(50), default=OrgRole.MEMBER.value, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    organization = relationship("Organization", back_populates="memberships")
    user = relationship("User", back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("organization_id", "user_id", name="uq_org_user_membership"),
    )
