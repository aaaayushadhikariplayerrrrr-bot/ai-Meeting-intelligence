from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yt_dlp

from app.core.config import settings
from app.services.media_validation import (
    validate_youtube_url,
    validate_media_content,
    validate_media_duration,
    MediaValidationError,
)
from app.services.audio_extraction import normalize_audio_for_asr
from app.services.media_metadata import MediaMetadata, extract_media_metadata, MetadataExtractionError


class YouTubeIngestionError(Exception):
    """Raised when YouTube ingestion fails."""
    pass


def extract_youtube_stream(
    youtube_url: str,
    output_dir: Path,
    confirmed_authorization: bool = False,
    raw_asset_id: Optional[str] = None,
    progress_hook: Optional[Any] = None,
) -> Tuple[Path, Path, Dict[str, Any], MediaMetadata, str]:
    """Ingest audio stream from user-provided YouTube URL using yt-dlp.
    
    Adheres strictly to legal & access parameters:
    - User must explicitly confirm authorization (`confirmed_authorization=True`).
    - Does not bypass DRM, paywalls, or login controls.
    - Inspects pre-download metadata to reject obviously over-limit videos early.
    - Performs authoritative FFprobe content and duration validation after download.
    - Cleans up partial and downloaded files on failure to prevent orphaned storage.
    - Extracts audio and standardizes to 16kHz mono WAV for ASR.
    
    Returns:
        Tuple[Path, Path, Dict[str, Any], MediaMetadata, str]:
            (raw_download_path, normalized_wav_path, video_info, normalized_metadata, sha256_checksum)
    """
    if not confirmed_authorization:
        raise YouTubeIngestionError(
            "User authorization confirmation is required to process external YouTube content."
        )

    is_valid, video_id, err = validate_youtube_url(youtube_url)
    if not is_valid:
        raise YouTubeIngestionError(err or "Invalid YouTube URL.")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_template = str(output_dir / f"yt_{video_id}_raw.%(ext)s")

    # 1. Pre-download metadata inspection: reject obviously over-limit videos before downloading streams
    ydl_opts_meta = {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": False,
        "noplaylist": True,
        "ignoreerrors": False,
        "geo_bypass": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl_meta:
            pre_info = ydl_meta.extract_info(youtube_url, download=False)
            if pre_info:
                pre_duration = pre_info.get("duration")
                if pre_duration is not None:
                    try:
                        dur_val = float(pre_duration)
                        if dur_val > settings.MAX_MEDIA_DURATION_SECONDS:
                            max_hrs = settings.MAX_MEDIA_DURATION_SECONDS / 3600.0
                            raise YouTubeIngestionError(
                                f"YouTube video duration ({dur_val:.1f}s) exceeds maximum allowed limit of "
                                f"{settings.MAX_MEDIA_DURATION_SECONDS:.1f}s ({max_hrs:.1f} hours)."
                            )
                    except ValueError:
                        pass
    except YouTubeIngestionError:
        raise
    except Exception:
        # If pre-download info extraction fails, allow download attempt to proceed where FFprobe will authoritatively validate
        pass

    # 2. Download audio stream
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": raw_template,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": False,
        "noplaylist": True,
        # Do not bypass DRM or restricted content
        "ignoreerrors": False,
        "geo_bypass": False,
    }

    if progress_hook:
        def _hook(d):
            if d.get("status") == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                downloaded = d.get("downloaded_bytes") or 0
                if total > 0:
                    try:
                        progress_hook((downloaded / total) * 100.0)
                    except Exception:
                        pass
        ydl_opts["progress_hooks"] = [_hook]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            if not info:
                raise YouTubeIngestionError("Failed to extract YouTube metadata.")

            downloaded_filename = ydl.prepare_filename(info)
            raw_media_path = Path(downloaded_filename)

            if not raw_media_path.exists():
                # Some extractors append different extensions
                matching = list(output_dir.glob(f"yt_{video_id}_raw.*"))
                if matching:
                    raw_media_path = matching[0]
                else:
                    raise YouTubeIngestionError("Downloaded YouTube file could not be located on disk.")

    except yt_dlp.utils.DownloadError as e:
        raise YouTubeIngestionError(f"YouTube download failed: {str(e)}")
    except YouTubeIngestionError:
        raise
    except Exception as e:
        raise YouTubeIngestionError(f"Unexpected error during YouTube ingestion: {str(e)}")

    # 3. Authoritative post-download content & duration inspection via FFprobe
    # Never trust YouTube metadata alone
    try:
        is_valid, err = validate_media_content(raw_media_path)
        if not is_valid:
            raise YouTubeIngestionError(f"Downloaded YouTube media failed content validation: {err}")

        ffprobe_meta = extract_media_metadata(raw_media_path)
        validated_duration = validate_media_duration(ffprobe_meta.duration_seconds)
    except Exception as e:
        # Clean up raw downloaded file and any partial files in output_dir
        if raw_media_path.exists():
            raw_media_path.unlink(missing_ok=True)
        for extra_file in output_dir.glob(f"yt_{video_id}_raw*"):
            extra_file.unlink(missing_ok=True)
        for part_file in output_dir.glob("*.part*"):
            part_file.unlink(missing_ok=True)
        if isinstance(e, YouTubeIngestionError):
            raise
        raise YouTubeIngestionError(f"YouTube media duration validation failed: {str(e)}")

    # 4. Extract video info dict with authoritative FFprobe duration
    video_info = {
        "id": info.get("id"),
        "title": info.get("title", f"YouTube Video {video_id}"),
        "description": info.get("description", ""),
        "duration": validated_duration,
        "uploader": info.get("uploader", "Unknown"),
        "webpage_url": info.get("webpage_url", youtube_url),
    }

    # 5. Normalize audio track to standard 16kHz mono WAV for ASR
    normalized_filename = (
        f"{raw_asset_id}_yt_{video_id}_normalized_16k.wav"
        if raw_asset_id
        else f"yt_{video_id}_normalized_16k.wav"
    )
    normalized_wav_path = output_dir / normalized_filename
    try:
        normalized_path, normalized_meta, checksum = normalize_audio_for_asr(
            raw_media_path,
            normalized_wav_path
        )
    except Exception as e:
        if raw_media_path.exists():
            raw_media_path.unlink(missing_ok=True)
        if normalized_wav_path.exists():
            normalized_wav_path.unlink(missing_ok=True)
        raise YouTubeIngestionError(f"Audio normalization failed: {str(e)}")

    return raw_media_path, normalized_path, video_info, normalized_meta, checksum
