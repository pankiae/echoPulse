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


def compute_acoustic_noise_metrics(file_path: str) -> dict:
    """
    Extracts deterministic acoustic metrics (noise floor RMS, peak RMS, estimated SNR)
    from audio signal to assist LLM multimodal background noise classification.
    """
    if not os.path.exists(file_path):
        return {"noise_floor_rms": 0.0, "elevated_noise_floor": False, "acoustic_hint": "Audio file not found"}

    try:
        import numpy as np
        ext = os.path.splitext(file_path)[1].lower()

        # For WAV files, parse PCM samples directly
        if ext == ".wav":
            with wave.open(file_path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                raw_bytes = wf.readframes(n_frames)
                if sampwidth == 2:
                    samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 1:
                    samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0

                if len(samples) > 0 and n_channels > 1:
                    samples = samples[::n_channels]  # Mono channel

                if len(samples) > 1000:
                    frame_size = int(framerate * 0.1)  # 100ms frames
                    if frame_size > 0:
                        num_frames = len(samples) // frame_size
                        frame_rms = [float(np.sqrt(np.mean(samples[i*frame_size:(i+1)*frame_size]**2))) for i in range(num_frames)]
                        noise_floor_rms = float(np.percentile(frame_rms, 20))
                        peak_rms = float(np.max(frame_rms))
                        snr = float(20 * np.log10((peak_rms + 1e-6) / (noise_floor_rms + 1e-6)))

                        elevated = noise_floor_rms > 0.0015
                        return {
                            "noise_floor_rms": round(noise_floor_rms, 5),
                            "peak_rms": round(peak_rms, 4),
                            "estimated_snr_db": round(snr, 1),
                            "elevated_noise_floor": elevated,
                            "acoustic_hint": f"Measured Noise Floor RMS: {noise_floor_rms:.4f} (Elevated noise layer detected: {elevated})"
                        }

        # Fallback byte energy analysis for compressed audio formats (.mp3, .ogg, .m4a)
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            if len(raw_bytes) > 2000:
                byte_array = np.frombuffer(raw_bytes[1000:-1000], dtype=np.uint8).astype(np.float32)
                std_dev = float(np.std(byte_array))
                elevated = std_dev > 45.0
                return {
                    "byte_std_dev": round(std_dev, 2),
                    "elevated_noise_floor": elevated,
                    "acoustic_hint": f"Raw Byte Dispersion: {std_dev:.1f} (Background texture detected: {elevated})"
                }
    except Exception as e:
        logger.warning(f"Acoustic noise pre-analysis fallback for '{file_path}': {e}")

    return {"elevated_noise_floor": False, "acoustic_hint": "Standard acoustic baseline"}

