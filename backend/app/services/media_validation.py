import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from app.core.config import settings

# Supported YouTube Hostnames
YOUTUBE_DOMAINS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com"
}


class MediaValidationError(Exception):
    """Raised when media validation fails."""
    pass


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension without the leading dot."""
    ext = Path(filename).suffix.lower()
    return ext[1:] if ext.startswith(".") else ext


def validate_file_extension(filename: str) -> Tuple[str, str]:
    """Validate filename against allowed audio and video extensions.
    
    Returns:
        Tuple[str, str]: (extension, media_category: 'AUDIO' or 'VIDEO')
    Raises:
        MediaValidationError: if extension is not supported.
    """
    ext = get_file_extension(filename)
    if not ext:
        raise MediaValidationError("Uploaded file has no extension.")

    if ext in settings.ALLOWED_AUDIO_EXTENSIONS:
        return ext, "AUDIO"
    elif ext in settings.ALLOWED_VIDEO_EXTENSIONS:
        return ext, "VIDEO"
    else:
        allowed = sorted(list(settings.ALLOWED_AUDIO_EXTENSIONS | settings.ALLOWED_VIDEO_EXTENSIONS))
        raise MediaValidationError(
            f"Unsupported file format '.{ext}'. Supported formats: {', '.join(allowed)}"
        )


def validate_media_content(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Inspect actual container/stream integrity using ffprobe rather than trusting client MIME types.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_reason)
    """
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=format_name:stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if proc.returncode != 0:
            return False, "File is not a valid audio or video container or is corrupted."

        output = proc.stdout.strip().splitlines()
        has_audio = any("audio" in line for line in output)
        has_video = any("video" in line for line in output)

        if not (has_audio or has_video):
            return False, "Media container does not contain any valid audio or video streams."

        return True, None
    except subprocess.TimeoutExpired:
        return False, "Media inspection timed out."
    except Exception as e:
        return False, f"Media inspection failed: {str(e)}"


def validate_media_duration(
    duration_seconds: Optional[float],
    max_duration: Optional[float] = None
) -> float:
    """Validate that media duration is present, finite, strictly positive,
    and does not exceed the configured maximum duration limit.

    Args:
        duration_seconds: Extracted duration in seconds.
        max_duration: Maximum allowed duration in seconds. Defaults to
                      settings.MAX_MEDIA_DURATION_SECONDS.

    Returns:
        float: Validated duration in seconds.

    Raises:
        MediaValidationError: If duration is missing, malformed, non-positive,
                              NaN, infinite, or exceeds max_duration.
    """
    limit = max_duration if max_duration is not None else settings.MAX_MEDIA_DURATION_SECONDS

    if duration_seconds is None:
        raise MediaValidationError("Media duration could not be determined or is missing.")

    try:
        dur = float(duration_seconds)
    except (ValueError, TypeError):
        raise MediaValidationError("Media duration value is malformed or invalid.")

    if math.isnan(dur) or math.isinf(dur):
        raise MediaValidationError("Media duration must be a finite number.")

    if dur <= 0.0:
        raise MediaValidationError(f"Media duration must be strictly positive (> 0s), got {dur:.2f}s.")

    if dur > limit:
        max_hrs = limit / 3600.0
        raise MediaValidationError(
            f"Media duration ({dur:.1f}s) exceeds maximum allowed limit of {limit:.1f}s ({max_hrs:.1f} hours)."
        )

    return dur


def validate_youtube_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validate user-provided YouTube URL and extract video ID.
    
    Rejects unsupported domains to prevent SSRF and arbitrary resource access.
    
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: (is_valid, video_id, error_message)
    """
    if not url or not isinstance(url, str):
        return False, None, "URL cannot be empty."

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False, None, "Invalid URL scheme. Only HTTP and HTTPS are permitted."

    hostname = (parsed.hostname or "").lower()
    if hostname not in YOUTUBE_DOMAINS:
        return False, None, f"Unsupported domain '{hostname}'. Only YouTube URLs are accepted."

    video_id = None
    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif "watch" in parsed.path:
        query_params = re.findall(r"v=([a-zA-Z0-9_-]{11})", parsed.query)
        if query_params:
            video_id = query_params[0]
    elif "embed" in parsed.path or "shorts" in parsed.path:
        match = re.search(r"/(?:embed|shorts)/([a-zA-Z0-9_-]{11})", parsed.path)
        if match:
            video_id = match.group(1)

    if not video_id or len(video_id) != 11:
        return False, None, "Could not extract a valid 11-character YouTube video ID from URL."

    return True, video_id, None
