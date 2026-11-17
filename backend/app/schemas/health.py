from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Schema for system health check status."""
    status: str = Field(..., description="Application health status, e.g. 'ok'")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="API Version")
    environment: str = Field(..., description="Current running environment")
    database: str = Field(..., description="Database connectivity status: 'connected' or 'disconnected'")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check"
    )
