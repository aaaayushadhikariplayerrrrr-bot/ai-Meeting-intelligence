"""Services package containing business logic, ingestion pipelines, and lifecycle services."""
from .media_validation import (
    validate_file_extension,
    validate_media_content,
    validate_youtube_url,
    MediaValidationError,
)
from .media_storage import (
    get_base_storage_dir,
    get_meeting_storage_dir,
    generate_asset_filename,
    compute_sha256,
    save_upload_stream,
    to_safe_relative_path,
    resolve_safe_storage_path,
    StorageError,
)
from .media_metadata import extract_media_metadata, MediaMetadata, MetadataExtractionError
from .audio_extraction import normalize_audio_for_asr, AudioExtractionError
from .youtube_ingestion import extract_youtube_stream, YouTubeIngestionError
from .job_manager import create_processing_job, update_processing_job
from .job_worker import (
    claim_job_atomically,
    get_next_pending_job_id,
    reconcile_stranded_jobs_on_startup,
    execute_claimed_job,
    run_single_job_cycle,
    JobWorker,
)
from .lifecycle import (
    delete_raw_media_asset,
    find_expired_media_assets,
    cleanup_expired_media_assets,
    MediaLifecycleError,
)
from .retention_scheduler import RetentionScheduler

__all__ = [
    "validate_file_extension",
    "validate_media_content",
    "validate_youtube_url",
    "MediaValidationError",
    "get_base_storage_dir",
    "get_meeting_storage_dir",
    "generate_asset_filename",
    "compute_sha256",
    "save_upload_stream",
    "to_safe_relative_path",
    "resolve_safe_storage_path",
    "StorageError",
    "extract_media_metadata",
    "MediaMetadata",
    "MetadataExtractionError",
    "normalize_audio_for_asr",
    "AudioExtractionError",
    "extract_youtube_stream",
    "YouTubeIngestionError",
    "create_processing_job",
    "update_processing_job",
    "claim_job_atomically",
    "get_next_pending_job_id",
    "reconcile_stranded_jobs_on_startup",
    "execute_claimed_job",
    "run_single_job_cycle",
    "JobWorker",
    "delete_raw_media_asset",
    "find_expired_media_assets",
    "cleanup_expired_media_assets",
    "MediaLifecycleError",
    "RetentionScheduler",
]
