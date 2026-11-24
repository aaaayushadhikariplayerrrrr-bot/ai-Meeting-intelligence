"""Pydantic schemas for Project management and meeting grouping."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, max_length=2000, description="Project description or goals")


class ProjectCreate(ProjectBase):
    slug: Optional[str] = Field(None, max_length=255, description="Custom slug (auto-generated if omitted)")


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    organization_id: str
    slug: str
    created_by_user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    meeting_count: int = 0
