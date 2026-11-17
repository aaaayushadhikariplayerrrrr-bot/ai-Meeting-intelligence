"""Automated Media Retention and Expired Asset Cleanup Scheduler.

Phase 6D.6:
- Provides a lightweight, non-blocking asyncio background scheduler integrated
  into the FastAPI lifespan.
- Runs an initial cleanup cycle at startup, then periodically at configured interval.
- Default retention period: 30 days (settings.RETENTION_DAYS).
- Default cleanup interval: 24 hours (settings.RETENTION_CLEANUP_INTERVAL_SECONDS).
- Offloads synchronous database and filesystem cleanup to a worker thread via asyncio.to_thread.
- Stops cleanly and deterministically during application shutdown.
- Ensures only one retention scheduler runs per application process.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.session import SessionLocal
from app.services.lifecycle import cleanup_expired_media_assets

logger = logging.getLogger(__name__)


class RetentionScheduler:
    """Lightweight asyncio background scheduler for automated media retention cleanup."""

    def __init__(
        self,
        cleanup_interval: Optional[float] = None,
        retention_days: Optional[int] = None,
    ):
        self.cleanup_interval = (
            cleanup_interval
            if cleanup_interval is not None
            else getattr(settings, "RETENTION_CLEANUP_INTERVAL_SECONDS", 86400.0)
        )
        self.retention_days = (
            retention_days
            if retention_days is not None
            else getattr(settings, "RETENTION_DAYS", 30)
        )
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """Start the background retention cleanup scheduler."""
        if self._is_running:
            logger.warning("RetentionScheduler is already running.")
            return

        self._stop_event.clear()
        self._is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"RetentionScheduler started (cleanup interval: {self.cleanup_interval}s, "
            f"retention: {self.retention_days} days)."
        )

    async def stop(self) -> None:
        """Signal the scheduler to stop and await termination."""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._is_running = False
        logger.info("RetentionScheduler stopped.")

    def _sync_cleanup(self) -> Dict[str, Any]:
        """Synchronously execute cleanup inside a dedicated database session."""
        db: Session = SessionLocal()
        try:
            return cleanup_expired_media_assets(db, retention_days=self.retention_days)
        except Exception as e:
            logger.exception(f"Unhandled error during retention cleanup cycle: {e}")
            return {"error": str(e)}
        finally:
            db.close()

    async def run_once(self) -> Dict[str, Any]:
        """Execute a single cleanup cycle offloaded to a worker thread without blocking asyncio."""
        return await asyncio.to_thread(self._sync_cleanup)

    async def _run_loop(self) -> None:
        """Periodic loop executing cleanup at startup and every cleanup_interval seconds."""
        logger.info("Executing initial startup media retention cleanup cycle.")
        try:
            await self.run_once()
        except Exception as e:
            logger.exception(f"Initial retention cleanup cycle failed: {e}")

        while not self._stop_event.is_set():
            try:
                # Wait for cleanup_interval or until stop_event is signaled
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.cleanup_interval)
                # If wait returned without TimeoutError, stop_event was signaled
                break
            except asyncio.TimeoutError:
                # Interval elapsed: trigger next periodic cycle
                pass

            if self._stop_event.is_set():
                break

            try:
                await self.run_once()
            except Exception as e:
                logger.exception(f"Periodic media retention cleanup cycle failed: {e}")
