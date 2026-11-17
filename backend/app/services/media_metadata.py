import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel


class MediaMetadata(BaseModel):
    """Normalized metadata structure extracted from media container."""
    format_name: str
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    bit_rate: Optional[int] = None
    # Audio properties
    has_audio: bool = False
    audio_codec: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    # Video properties
    has_video: bool = False
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None


class MetadataExtractionError(Exception):
    """Raised when ffprobe metadata extraction fails."""
    pass


def extract_media_metadata(file_path: Path) -> MediaMetadata:
    """Execute ffprobe and parse container & stream properties into MediaMetadata.
    
    Raises:
        MetadataExtractionError: if ffprobe fails or output cannot be parsed.
    """
    ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            raise MetadataExtractionError(
                f"ffprobe exited with code {proc.returncode}: {proc.stderr.strip()}"
            )

        data: Dict[str, Any] = json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        raise MetadataExtractionError("Metadata extraction timed out.")
    except json.JSONDecodeError as e:
        raise MetadataExtractionError(f"Failed to parse ffprobe JSON output: {str(e)}")
    except Exception as e:
        raise MetadataExtractionError(f"Unexpected error running ffprobe: {str(e)}")

    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration_val = format_info.get("duration")
    duration_seconds = None
    if duration_val is not None:
        try:
            duration_seconds = float(duration_val)
        except (ValueError, TypeError):
            duration_seconds = None

    # Fallback to streams duration if format duration was absent
    if duration_seconds is None:
        for stream in streams:
            s_dur = stream.get("duration")
            if s_dur is not None:
                try:
                    duration_seconds = float(s_dur)
                    break
                except (ValueError, TypeError):
                    continue

    size_val = format_info.get("size")
    file_size_bytes = int(size_val) if size_val is not None else None

    bit_rate_val = format_info.get("bit_rate")
    bit_rate = int(bit_rate_val) if bit_rate_val is not None else None

    # Audio stream inspection
    has_audio = False
    audio_codec = None
    sample_rate = None
    channels = None

    # Video stream inspection
    has_video = False
    video_codec = None
    width = None
    height = None
    fps = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "audio" and not has_audio:
            has_audio = True
            audio_codec = stream.get("codec_name")
            sr_val = stream.get("sample_rate")
            sample_rate = int(sr_val) if sr_val is not None else None
            ch_val = stream.get("channels")
            channels = int(ch_val) if ch_val is not None else None
        elif codec_type == "video" and not has_video:
            has_video = True
            video_codec = stream.get("codec_name")
            width = stream.get("width")
            height = stream.get("height")
            # Calculate FPS from r_frame_rate or avg_frame_rate
            rate_str = stream.get("r_frame_rate") or stream.get("avg_frame_rate")
            if rate_str and "/" in rate_str:
                num, denom = rate_str.split("/")
                if float(denom) > 0:
                    fps = round(float(num) / float(denom), 2)

    return MediaMetadata(
        format_name=format_info.get("format_name", "unknown"),
        duration_seconds=duration_seconds,
        file_size_bytes=file_size_bytes,
        bit_rate=bit_rate,
        has_audio=has_audio,
        audio_codec=audio_codec,
        sample_rate=sample_rate,
        channels=channels,
        has_video=has_video,
        video_codec=video_codec,
        width=width,
        height=height,
        fps=fps
    )
