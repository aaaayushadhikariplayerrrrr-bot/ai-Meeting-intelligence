"""Project model for tenant-isolated meeting grouping."""
from sqlalchemy import Column, String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database.base import Base
from app.models.base import UUIDMixin, TimestampMixin


class Project(UUIDMixin, TimestampMixin, Base):
    """Project entity representing an organizational initiative or workspace grouping."""
    __tablename__ = "projects"

    organization_id = Column(
        String(36),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_by_user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    meetings = relationship("Meeting", back_populates="project")
    created_by_user = relationship(
        "User",
        foreign_keys=[created_by_user_id]
    )

    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_projects_organization_slug"),
    )
