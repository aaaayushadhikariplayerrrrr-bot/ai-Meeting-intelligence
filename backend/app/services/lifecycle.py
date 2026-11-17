import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.config import settings
from app.models.meeting import MediaAsset, Meeting, ProcessingJob, JobType, JobStatus
from app.models.memory import AuditLog
from app.services.media_storage import resolve_safe_storage_path

logger = logging.getLogger(__name__)


class MediaLifecycleError(Exception):
    """Raised when media lifecycle or deletion operations fail."""
    pass


def _matches_raw_asset_id(raw_id: str, candidate_filename: Optional[str], candidate_path: Optional[str]) -> bool:
    """Verify if candidate filename or storage path explicitly associates with raw_id via exact token boundary."""
    if not raw_id:
        return False

    # Normalize slashes for cross-platform POSIX and Windows compatibility
    norm_path = (candidate_path or "").replace("\\", "/")
    path_filename = norm_path.rsplit("/", 1)[-1] if "/" in norm_path else norm_path
    target_filename = candidate_filename or path_filename

    if not target_filename:
        return False

    # Standard convention: exact token prefix `{raw_id}_`
    # Examples:
    # "{raw_asset.id}_normalized_16k.wav"
    # "{raw_asset.id}_yt_{video_id}_normalized_16k.wav"
    # Prevents "asset-1" from matching "asset-10_normalized_16k.wav"
    if target_filename.startswith(f"{raw_id}_"):
        return True

    # Exact stem match (e.g. "{raw_id}.wav") or "{raw_id}_" stem prefix
    stem = target_filename.split(".", 1)[0]
    if stem == raw_id or stem.startswith(f"{raw_id}_"):
        return True

    return False


def is_associated_normalized_asset(
    raw_asset: MediaAsset,
    candidate: MediaAsset,
    db: Session
) -> bool:
    """Explicitly verify whether candidate normalized asset is associated with raw_asset.
    
    Verification rules:
    1. Tenant & Meeting Scoping: candidate MUST belong to the exact same meeting (candidate.meeting_id == raw_asset.meeting_id).
    2. Asset Identity: candidate cannot be the raw_asset itself (candidate.id != raw_asset.id).
    3. Type Confirmation: candidate must be an audio asset representing normalized ASR audio (candidate.media_type == "NORMALIZED_AUDIO").
    4. Exact Association Linkage:
       - Exact token prefix matching: candidate's file_name or storage_path explicitly begins with `{raw_asset.id}_`
         (standard uploads: `{raw_asset.id}_normalized_16k.wav`, YouTube: `{raw_asset.id}_yt_{video_id}_normalized_16k.wav`).
       - Exclusivity check: If candidate matches another raw asset's ID in this meeting, candidate is NOT associated with this raw_asset.
       - Sole raw asset in meeting: If candidate does not bear any raw asset token (e.g. legacy/unlinked test fixtures),
         candidate is associated with raw_asset ONLY IF raw_asset is the sole raw asset for that meeting. If there are
         multiple raw assets without explicit ID linkage, candidate cannot be conclusively attributed to raw_asset,
         preventing unintended deletion.
    """
    if candidate.id == raw_asset.id:
        return False

    if candidate.meeting_id != raw_asset.meeting_id:
        return False

    if candidate.media_type != "NORMALIZED_AUDIO":
        return False

    raw_id = str(raw_asset.id)
    cand_file = candidate.file_name or ""
    cand_path = candidate.storage_path or ""

    # Check exact token association match
    if _matches_raw_asset_id(raw_id, cand_file, cand_path):
        return True

    # Check if candidate explicitly matches another raw asset in this meeting
    meeting_raw_assets = (
        db.query(MediaAsset)
        .filter(
            MediaAsset.meeting_id == raw_asset.meeting_id,
            MediaAsset.media_type != "NORMALIZED_AUDIO",
            MediaAsset.id != raw_asset.id
        )
        .all()
    )
    for other_raw in meeting_raw_assets:
        if _matches_raw_asset_id(str(other_raw.id), cand_file, cand_path):
            return False

    # If no other raw assets exist in this meeting and candidate has no provable raw association,
    # associate with the sole raw asset
    if len(meeting_raw_assets) == 0:
        return True

    # Conservative: cannot prove association in multi-asset meeting, do not delete
    return False


def delete_raw_media_asset(
    db: Session,
    media_asset_id: str,
    user_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    delete_normalized: bool = True
) -> MediaAsset:
    """Safely delete media file from disk and record an immutable audit log.
    
    If delete_normalized is True and the asset is a raw asset (not NORMALIZED_AUDIO),
    any explicitly verified associated normalized audio asset(s) for the meeting are
    also safely deleted and distinct audit log entries are recorded.
    
    IMPORTANT:
    - Scoped strictly to the same meeting and verified associated assets.
    - Idempotent: safe to invoke repeatedly on already deleted assets.
    - Application-level unlinking (no false claims of SSD forensic wipe).
    - If physical file is missing from disk, updates DB flag safely without error.
    - Creates immutable append-only AuditLog records for each deleted asset.
    """
    asset = db.get(MediaAsset, media_asset_id)
    if not asset:
        raise MediaLifecycleError(f"MediaAsset with ID {media_asset_id} not found.")

    meeting = db.get(Meeting, asset.meeting_id)
    org_id = meeting.organization_id if meeting else None

    # 1. Identify associated normalized assets if requested
    associated_normalized: List[MediaAsset] = []
    if delete_normalized and asset.media_type != "NORMALIZED_AUDIO":
        candidate_normalized = (
            db.query(MediaAsset)
            .filter(
                MediaAsset.meeting_id == asset.meeting_id,
                MediaAsset.media_type == "NORMALIZED_AUDIO",
                MediaAsset.id != asset.id
            )
            .all()
        )
        for cand in candidate_normalized:
            if is_associated_normalized_asset(asset, cand, db):
                associated_normalized.append(cand)

    # 2. Check idempotency: if primary asset and all associated assets are already deleted, return early
    all_norm_already_deleted = all(n.is_raw_deleted for n in associated_normalized)
    if asset.is_raw_deleted and all_norm_already_deleted:
        return asset

    # 3. Safely delete primary asset file if not already deleted
    if not asset.is_raw_deleted:
        try:
            physical_path = resolve_safe_storage_path(asset.storage_path)
        except Exception as e:
            raise MediaLifecycleError(f"Could not resolve safe storage path: {str(e)}")

        file_existed = False
        if physical_path.exists():
            file_existed = True
            try:
                physical_path.unlink()
            except OSError as e:
                raise MediaLifecycleError(f"Operating system failed to delete file: {str(e)}")

        asset.is_raw_deleted = True

        raw_audit = AuditLog(
            organization_id=org_id,
            user_id=user_id,
            action="DELETE_RAW_AUDIO" if asset.media_type != "NORMALIZED_AUDIO" else "DELETE_NORMALIZED_AUDIO",
            entity_type="MediaAsset",
            entity_id=asset.id,
            details=json.dumps({
                "file_name": asset.file_name,
                "storage_path": asset.storage_path,
                "asset_role": "ORIGINAL_RAW" if asset.media_type != "NORMALIZED_AUDIO" else "NORMALIZED_AUDIO",
                "media_type": asset.media_type,
                "status": "deleted" if file_existed else "file_not_found_marked_deleted",
                "deletion_type": "application_unlink"
            }),
            ip_address=client_ip
        )
        db.add(raw_audit)

    # 4. Safely delete associated normalized asset(s)
    for norm_asset in associated_normalized:
        if not norm_asset.is_raw_deleted:
            try:
                norm_physical_path = resolve_safe_storage_path(norm_asset.storage_path)
            except Exception as e:
                logger.warning(f"Could not resolve safe storage path for normalized asset {norm_asset.id}: {e}")
                norm_physical_path = None

            norm_file_existed = False
            if norm_physical_path and norm_physical_path.exists():
                norm_file_existed = True
                try:
                    norm_physical_path.unlink()
                except OSError as e:
                    raise MediaLifecycleError(f"Operating system failed to delete normalized audio file: {str(e)}")

            norm_asset.is_raw_deleted = True

            norm_audit = AuditLog(
                organization_id=org_id,
                user_id=user_id,
                action="DELETE_NORMALIZED_AUDIO",
                entity_type="MediaAsset",
                entity_id=norm_asset.id,
                details=json.dumps({
                    "file_name": norm_asset.file_name,
                    "storage_path": norm_asset.storage_path,
                    "asset_role": "NORMALIZED_AUDIO",
                    "source_raw_asset_id": asset.id,
                    "media_type": norm_asset.media_type,
                    "status": "deleted" if norm_file_existed else "file_not_found_marked_deleted",
                    "deletion_type": "application_unlink"
                }),
                ip_address=client_ip
            )
            db.add(norm_audit)

    db.commit()
    db.refresh(asset)
    return asset


def find_expired_media_assets(
    db: Session,
    retention_days: int = settings.RETENTION_DAYS,
    cutoff_date: Optional[datetime] = None,
) -> List[MediaAsset]:
    """Find media assets older than retention_days that have not yet had raw media deleted."""
    if cutoff_date is None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    stmt = (
        select(MediaAsset)
        .where(
            MediaAsset.created_at < cutoff_date,
            MediaAsset.is_raw_deleted.is_(False)
        )
        .order_by(MediaAsset.created_at.asc())
    )
    return list(db.execute(stmt).scalars().all())


def cleanup_expired_media_assets(
    db: Session,
    retention_days: Optional[int] = None,
    client_ip: Optional[str] = "retention_scheduler",
) -> Dict[str, Any]:
    """Find and safely delete media assets exceeding the retention policy.

    Safe Deletion Guarantees:
    - Rechecks expiration immediately before deleting each asset.
    - Preserves assets associated with active media-consuming jobs (PENDING or IN_PROGRESS).
    - Preserves unassociated or uncertain normalized audio assets.
    - Safely unlinks physical files, handles missing files gracefully, and writes immutable AuditLog records.
    - Does not delete transcript, segment, summary, or meeting intelligence records.
    - Isolates failures: error on one asset does not abort cleanup of remaining eligible assets.
    """
    days = retention_days if retention_days is not None else getattr(settings, "RETENTION_DAYS", 30)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    candidate_assets = find_expired_media_assets(db, retention_days=days, cutoff_date=cutoff_date)

    stats: Dict[str, Any] = {
        "examined_count": len(candidate_assets),
        "deleted_raw_count": 0,
        "deleted_normalized_count": 0,
        "skipped_active_jobs_count": 0,
        "skipped_already_deleted_count": 0,
        "skipped_not_expired_count": 0,
        "skipped_uncertain_association_count": 0,
        "error_count": 0,
        "errors": [],
    }

    media_consuming_job_types = [
        JobType.TRANSCRIPTION.value,
        JobType.INGESTION.value,
        JobType.DIARIZATION.value,
    ]

    for candidate in candidate_assets:
        try:
            # Re-fetch asset from database
            asset = db.get(MediaAsset, candidate.id)
            if not asset:
                continue

            # 1. Idempotency check: skip if already deleted
            if asset.is_raw_deleted:
                stats["skipped_already_deleted_count"] += 1
                continue

            # 2. Expiration recheck: verify created_at vs current cutoff
            asset_created_at = asset.created_at
            if asset_created_at.tzinfo is None:
                asset_created_at = asset_created_at.replace(tzinfo=timezone.utc)
            if asset_created_at >= cutoff_date:
                stats["skipped_not_expired_count"] += 1
                continue

            # 3. Active jobs check: do not delete if any media-consuming job is PENDING or IN_PROGRESS
            active_jobs_count = (
                db.query(ProcessingJob)
                .filter(
                    ProcessingJob.meeting_id == asset.meeting_id,
                    ProcessingJob.job_type.in_(media_consuming_job_types),
                    ProcessingJob.status.in_([JobStatus.PENDING.value, JobStatus.IN_PROGRESS.value]),
                )
                .count()
            )
            if active_jobs_count > 0:
                logger.info(
                    f"Skipping retention cleanup for asset {asset.id} in meeting {asset.meeting_id}: "
                    f"{active_jobs_count} active media-consuming job(s) in progress."
                )
                stats["skipped_active_jobs_count"] += 1
                continue

            meeting = db.get(Meeting, asset.meeting_id)
            user_id = meeting.created_by_user_id if meeting else None

            # 4. Route by media type:
            if asset.media_type != "NORMALIZED_AUDIO":
                # Primary/Raw Asset: delete_raw_media_asset unlinks raw + explicitly verified associated normalized
                pre_norm = (
                    db.query(MediaAsset)
                    .filter(
                        MediaAsset.meeting_id == asset.meeting_id,
                        MediaAsset.media_type == "NORMALIZED_AUDIO",
                        MediaAsset.is_raw_deleted.is_(False),
                        MediaAsset.id != asset.id
                    )
                    .all()
                )
                norm_to_delete = [c for c in pre_norm if is_associated_normalized_asset(asset, c, db)]

                delete_raw_media_asset(
                    db=db,
                    media_asset_id=asset.id,
                    user_id=user_id,
                    client_ip=client_ip,
                    delete_normalized=True
                )
                stats["deleted_raw_count"] += 1
                stats["deleted_normalized_count"] += len(norm_to_delete)

            else:
                # Standalone or remaining normalized audio asset
                meeting_raw_assets = (
                    db.query(MediaAsset)
                    .filter(
                        MediaAsset.meeting_id == asset.meeting_id,
                        MediaAsset.media_type != "NORMALIZED_AUDIO",
                    )
                    .all()
                )
                non_deleted_raw = [r for r in meeting_raw_assets if not r.is_raw_deleted]

                # If any raw asset in the meeting is still un-deleted, preserve normalized audio
                if non_deleted_raw:
                    logger.info(
                        f"Preserving normalized asset {asset.id} in meeting {asset.meeting_id}: "
                        f"meeting still has {len(non_deleted_raw)} non-deleted raw asset(s)."
                    )
                    stats["skipped_uncertain_association_count"] += 1
                    continue

                if len(meeting_raw_assets) == 0:
                    # Direct normalized audio asset with no raw counterparts
                    delete_raw_media_asset(
                        db=db,
                        media_asset_id=asset.id,
                        user_id=user_id,
                        client_ip=client_ip,
                        delete_normalized=False
                    )
                    stats["deleted_normalized_count"] += 1

                elif len(meeting_raw_assets) == 1:
                    # Sole raw asset in meeting, association conclusively established
                    delete_raw_media_asset(
                        db=db,
                        media_asset_id=asset.id,
                        user_id=user_id,
                        client_ip=client_ip,
                        delete_normalized=False
                    )
                    stats["deleted_normalized_count"] += 1

                else:
                    # Multi-raw meeting: must match exact token prefix of one of the deleted raw assets
                    matched_raw = [
                        r for r in meeting_raw_assets
                        if _matches_raw_asset_id(str(r.id), asset.file_name, asset.storage_path)
                    ]
                    if len(matched_raw) == 1:
                        delete_raw_media_asset(
                            db=db,
                            media_asset_id=asset.id,
                            user_id=user_id,
                            client_ip=client_ip,
                            delete_normalized=False
                        )
                        stats["deleted_normalized_count"] += 1
                    else:
                        # Uncertain or unassociated: preserve to prevent unintended loss!
                        logger.warning(
                            f"Preserving normalized asset {asset.id} in meeting {asset.meeting_id}: "
                            f"cannot conclusively attribute to a specific raw asset among multiple raw assets."
                        )
                        stats["skipped_uncertain_association_count"] += 1
                        continue

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete expired media asset {candidate.id}: {str(e)}")
            stats["error_count"] += 1
            stats["errors"].append({"asset_id": candidate.id, "error": str(e)})

    logger.info(
        f"Media retention cleanup cycle finished: "
        f"examined={stats['examined_count']}, "
        f"deleted_raw={stats['deleted_raw_count']}, "
        f"deleted_normalized={stats['deleted_normalized_count']}, "
        f"skipped_active_jobs={stats['skipped_active_jobs_count']}, "
        f"skipped_uncertain={stats['skipped_uncertain_association_count']}, "
        f"errors={stats['error_count']}."
    )
    return stats
