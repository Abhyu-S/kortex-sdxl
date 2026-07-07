"""
SQLAlchemy ORM models for the Kortex task engine.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base


class Task(Base):
    """Represents an asynchronous image-generation task."""

    __tablename__ = "tasks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    status = Column(
        String(20),
        nullable=False,
        default="PENDING",
        index=True,
    )
    result_url = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<Task id={self.id} status={self.status}>"
