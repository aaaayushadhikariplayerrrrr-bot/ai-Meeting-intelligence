from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class MediaAssetResponse(BaseModel):
    id: str
    meeting_id: str
    file_name: str
    storage_path: str
    media_type: str
    mime_type: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    checksum_sha256: Optional[str] = None
    is_raw_deleted: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProcessingJobResponse(BaseModel):
    id: str
    meeting_id: str
    job_type: str
    status: str
    progress: float
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    attempt_count: int = 0
    max_attempts: int = 2
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class MeetingResponse(BaseModel):
    id: str
    organization_id: str
    title: str
    description: Optional[str] = None
    input_source_type: str
    source_url_or_reference: Optional[str] = None
    status: str
    created_at: datetime
    media_assets: List[MediaAssetResponse] = []
    processing_jobs: List[ProcessingJobResponse] = []

    model_config = ConfigDict(from_attributes=True)


class YouTubeIngestRequest(BaseModel):
    url: str = Field(..., description="Valid YouTube watch/share URL")
    title: Optional[str] = Field(None, description="Optional custom title for the meeting")
    description: Optional[str] = Field(None, description="Optional description")
    confirmed_authorization: bool = Field(
        False,
        description="Explicit confirmation that user has legal authorization to process this content"
    )


class IngestResponse(BaseModel):
    meeting_id: str
    job_id: str
    status: str
    message: str
    media_asset_id: Optional[str] = None
