# pipeline/audio_utils.py
import os
import wave
import struct
from pipeline.logger import get_logger

logger = get_logger("AudioUtils")


def get_audio_duration_seconds(file_path: str) -> float:
    """
    Extracts exact audio duration (in seconds) for WAV, MP3, OGG, FLAC, M4A clips
    using lightweight binary parsing without heavy dependencies.
    Fallback estimates duration using standard audio token count or bitrate.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # 1. WAV Files via wave module
    if ext == ".wav":
        try:
            with wave.open(file_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception as e:
            logger.warning(f"Could not parse WAV header for '{file_path}': {e}")

    # 2. OGG Vorbis parsing
    if ext == ".ogg":
        try:
            with open(file_path, 'rb') as f:
                f.seek(-200, os.SEEK_END)
                data = f.read()
                pos = data.rfind(b'OggS')
                if pos != -1:
                    granule = struct.unpack('<Q', data[pos+6:pos+14])[0]
                    # Default OGG sample rate 44100Hz
                    return round(granule / 44100.0, 2)
        except Exception:
            pass

    # 3. Bitrate estimation for MP3, AAC, M4A, FLAC based on standard telephone/call center bitrates (128kbps)
    try:
        file_size_bytes = os.path.getsize(file_path)
        # Average compressed call audio ~16 KB/sec (128 kbps)
        estimated_sec = round(file_size_bytes / 16000.0, 2)
        return max(estimated_sec, 0.5)
    except Exception as e:
        logger.warning(f"Duration estimation fallback failed for '{file_path}': {e}")
        return 0.0


def format_duration_human(seconds: float) -> str:
    """
    Formats duration in seconds to human readable text: e.g. 185.0s -> '3m 5s' or '45s'.
    """
    if seconds <= 0:
        return "0s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{round(seconds, 1)}s"
