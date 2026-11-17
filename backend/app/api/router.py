from fastapi import APIRouter
from app.api.endpoints import health, media, transcription, auth, intelligence, memory, cross_meeting

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(media.router, prefix="/media", tags=["Media Ingestion"])
api_router.include_router(transcription.router, tags=["Transcription"])
api_router.include_router(intelligence.router, tags=["Intelligence"])
api_router.include_router(cross_meeting.router, tags=["Cross-Meeting Intelligence"])
api_router.include_router(cross_meeting.router, prefix="/intelligence", tags=["Cross-Meeting Intelligence"])
api_router.include_router(memory.router, prefix="/memory", tags=["Organizational Memory"])
