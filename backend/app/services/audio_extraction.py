import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from app.services.media_metadata import extract_media_metadata, MediaMetadata
from app.services.media_storage import compute_sha256


class AudioExtractionError(Exception):
    """Raised when audio extraction or normalization fails."""
    pass


def normalize_audio_for_asr(
    input_media_path: Path,
    output_wav_path: Path
) -> Tuple[Path, MediaMetadata, str]:
    """Extract audio from input file (audio or video) and normalize to ASR standard:
    
    Standards:
        - Container: WAV
        - Codec: pcm_s16le (16-bit uncompressed PCM)
        - Sample Rate: 16,000 Hz (16 kHz)
        - Channels: 1 (Mono)
    
    Returns:
        Tuple[Path, MediaMetadata, str]: (output_path, metadata, sha256_checksum)
    Raises:
        AudioExtractionError: if ffmpeg fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",               # Overwrite output file if exists
        "-i", str(input_media_path),
        "-vn",              # Disable video recording
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16 kHz sample rate
        "-ac", "1",         # Mono channel
        str(output_wav_path)
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180     # Up to 3 minutes for large meeting extractions
        )
        if proc.returncode != 0:
            raise AudioExtractionError(
                f"FFmpeg audio extraction failed (code {proc.returncode}): {proc.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("FFmpeg audio extraction timed out.")
    except Exception as e:
        raise AudioExtractionError(f"Unexpected error running FFmpeg: {str(e)}")

    if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
        raise AudioExtractionError("Normalized audio file was not generated or is empty.")

    metadata = extract_media_metadata(output_wav_path)
    checksum = compute_sha256(output_wav_path)

    return output_wav_path, metadata, checksum
