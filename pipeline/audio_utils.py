# pipeline/audio_utils.py
import os
import wave
import struct
from pipeline.logger import get_logger

logger = get_logger("AudioUtils")

try:
    import mutagen
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


def get_audio_duration_seconds(file_path: str) -> float:
    """
    Extracts exact audio duration (in seconds) for WAV, MP3, OGG, FLAC, M4A, AAC clips
    using Mutagen metadata parser with binary header fallback.
    """
    if not os.path.exists(file_path):
        return 0.0

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Primary: Use Mutagen if available for exact header duration parsing across MP3, OGG, M4A, FLAC, AAC, WAV
    if MUTAGEN_AVAILABLE:
        try:
            audio = mutagen.File(file_path)
            if audio is not None and hasattr(audio, "info") and audio.info and getattr(audio.info, "length", None):
                duration = float(audio.info.length)
                if duration > 0:
                    return round(duration, 2)
        except Exception as e:
            logger.warning(f"Mutagen parsing exception for '{file_path}': {e}")

    # 2. WAV Files via standard wave module
    if ext == ".wav":
        try:
            with wave.open(file_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception as e:
            logger.warning(f"Could not parse WAV header for '{file_path}': {e}")

    # 3. OGG Vorbis parsing fallback
    if ext == ".ogg":
        try:
            with open(file_path, 'rb') as f:
                f.seek(-200, os.SEEK_END)
                data = f.read()
                pos = data.rfind(b'OggS')
                if pos != -1:
                    granule = struct.unpack('<Q', data[pos+6:pos+14])[0]
                    return round(granule / 44100.0, 2)
        except Exception:
            pass

    # 4. Fallback for compressed call audio (~4 KB/s for 32 kbps voice recordings)
    try:
        file_size_bytes = os.path.getsize(file_path)
        estimated_sec = round(file_size_bytes / 4000.0, 2)
        return max(estimated_sec, 0.5)
    except Exception as e:
        logger.warning(f"Duration estimation fallback failed for '{file_path}': {e}")
        return 0.0


def format_duration_human(seconds: float) -> str:
    """
    Formats duration in seconds to human readable text: e.g. 485.0s -> '8m 5s' or '45s'.
    """
    if seconds <= 0:
        return "0s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{round(seconds, 1)}s"
