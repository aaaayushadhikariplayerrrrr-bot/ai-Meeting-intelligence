import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime


def utc_now() -> datetime:
    """Return timezone-aware UTC current time."""
    return datetime.now(timezone.utc)


class UUIDMixin:
    """Mixin that adds a 36-character UUID primary key initialized on instantiation."""
    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True
    )

    def __init__(self, *args, **kwargs):
        if "id" not in kwargs:
            kwargs["id"] = str(uuid.uuid4())
        super().__init__(*args, **kwargs)


class TimestampMixin:
    """Mixin that adds timezone-aware created_at and updated_at timestamps."""
    created_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
