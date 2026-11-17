from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.transcription.gpu_manager import configure_cpu_resources
from app.api.router import api_router
from app.api.endpoints import health
from app.core.config import settings
from app.database.session import SessionLocal
from app.services.job_worker import reconcile_stranded_jobs_on_startup, JobWorker
from app.services.retention_scheduler import RetentionScheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Hardware & CPU Resource Allocation (Phase 9.3)
    # Bound OpenMP, BLAS, and PyTorch threads strictly once before background tasks or parallel compute
    configure_cpu_resources()

    # 1. Conservative startup recovery: reconcile abandoned IN_PROGRESS jobs
    db = SessionLocal()
    try:
        reconcile_stranded_jobs_on_startup(db)
    finally:
        db.close()

    # 2. Start exactly one background job worker
    poll_interval = getattr(settings, "JOB_POLL_INTERVAL_SECONDS", 2.0)
    worker = JobWorker(poll_interval=poll_interval)
    await worker.start()
    app.state.job_worker = worker

    # 3. Start exactly one background media retention scheduler
    retention_scheduler = None
    if getattr(settings, "ENABLE_RETENTION_SCHEDULER", True):
        retention_scheduler = RetentionScheduler(
            cleanup_interval=getattr(settings, "RETENTION_CLEANUP_INTERVAL_SECONDS", 86400.0),
            retention_days=getattr(settings, "RETENTION_DAYS", 30),
        )
        await retention_scheduler.start()
        app.state.retention_scheduler = retention_scheduler

    yield

    # 4. Graceful shutdown: stop retention scheduler and worker cleanly
    if retention_scheduler:
        await retention_scheduler.stop()
    await worker.stop()


is_production = (settings.ENVIRONMENT or "").strip().lower() == "production"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Privacy-aware, multilingual AI Meeting Intelligence and Organizational Memory System API.",
    lifespan=lifespan,
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else f"{settings.API_V1_STR}/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root-level health endpoint (as explicitly requested: /health)
app.include_router(health.router)

# Versioned API routes (/api/v1/...)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root():
    is_prod = (settings.ENVIRONMENT or "").strip().lower() == "production"
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": None if is_prod else "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
