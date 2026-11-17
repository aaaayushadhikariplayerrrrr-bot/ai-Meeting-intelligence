import hashlib
import os
import uuid
from pathlib import Path
from typing import BinaryIO, Tuple
from app.core.config import settings


class StorageError(Exception):
    """Raised when file storage operations fail."""
    pass


def get_base_storage_dir() -> Path:
    """Return the absolute path to the root media storage directory, creating it if needed."""
    storage_path = Path(settings.STORAGE_DIR).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    return storage_path


def get_meeting_storage_dir(meeting_id: str) -> Path:
    """Resolve and create meeting-specific secure storage folder under STORAGE_DIR.
    
    Prevents path traversal attacks.
    """
    base_dir = get_base_storage_dir()
    # Normalize meeting_id as safe alphanumeric/hyphen string
    clean_meeting_id = str(uuid.UUID(meeting_id))
    meeting_dir = (base_dir / clean_meeting_id).resolve()

    # Verify that target path remains inside base_dir
    if not str(meeting_dir).startswith(str(base_dir)):
        raise StorageError("Invalid storage path traversal attempt detected.")

    meeting_dir.mkdir(parents=True, exist_ok=True)
    return meeting_dir


def generate_asset_filename(extension: str) -> str:
    """Generate a secure, random UUID-based filename rather than trusting user input."""
    clean_ext = extension.lstrip(".").lower()
    return f"{uuid.uuid4()}.{clean_ext}"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum in streaming 64KB blocks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_upload_stream(
    stream: BinaryIO,
    destination_path: Path,
    max_bytes: int = settings.MAX_UPLOAD_SIZE_BYTES
) -> Tuple[int, str]:
    """Stream incoming upload file to destination path, enforcing size limits and computing checksum.
    
    Returns:
        Tuple[int, str]: (bytes_written, sha256_checksum)
    Raises:
        StorageError: if max_bytes is exceeded.
    """
    bytes_written = 0
    sha256_hash = hashlib.sha256()

    try:
        with open(destination_path, "wb") as out_f:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    out_f.close()
                    if destination_path.exists():
                        destination_path.unlink()
                    limit_mb = max_bytes // (1024 * 1024)
                    raise StorageError(f"File size exceeds maximum permitted upload limit of {limit_mb} MB.")
                
                sha256_hash.update(chunk)
                out_f.write(chunk)
    except StorageError:
        raise
    except Exception as e:
        if destination_path.exists():
            destination_path.unlink()
        raise StorageError(f"Failed to write file to disk: {str(e)}")

    return bytes_written, sha256_hash.hexdigest()


def to_safe_relative_path(abs_path: Path) -> str:
    """Convert absolute server filesystem path to a safe, relative storage key.
    
    Ensures internal filesystem paths are never leaked in API responses.
    """
    base_dir = get_base_storage_dir()
    try:
        relative = abs_path.resolve().relative_to(base_dir)
        return f"storage/{relative.as_posix()}"
    except ValueError:
        return f"storage/{abs_path.name}"


def resolve_safe_storage_path(safe_relative_path: str) -> Path:
    """Resolve a relative storage key back to a validated absolute path inside STORAGE_DIR.
    
    Prevents path traversal.
    """
    base_dir = get_base_storage_dir()
    clean_subpath = safe_relative_path.replace("storage/", "").lstrip("/\\")
    resolved = (base_dir / clean_subpath).resolve()
    if not str(resolved).startswith(str(base_dir)):
        raise StorageError("Invalid storage path traversal attempt detected.")
    return resolved
