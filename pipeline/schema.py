# pipeline/schema.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BackgroundNoiseSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


class AudioAnalysisResult(BaseModel):
    emotional_tone: EmotionalTone = Field(..., description="Primary emotional tone expressed by customer")
    emotional_intensity: EmotionalIntensity = Field(..., description="Strength of emotional tone")
    background_noise_present: bool = Field(..., description="Audible background non-speech sound")
    background_noise_type: str = Field(default="", description="Concise description of dominant background noise (e.g., 'office chatter', 'TV', 'sharp static', 'traffic', 'siren', or empty string if no noise)")
    background_noise_severity: BackgroundNoiseSeverity = Field(..., description="Impact of noise on call clarity")
    audio_quality: AudioQuality = Field(..., description="Technical audio quality independent of emotion")
    speaker_overlap_present: bool = Field(..., description="Multiple speakers talking at once")
    long_silence_present: bool = Field(..., description="Unusual dead air / continuous silence period (>4 seconds)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")


class BatchFileResult(BaseModel):
    filename: str
    status: str = "success"  # "success" or "failed"
    error: Optional[str] = None
    prediction: Optional[AudioAnalysisResult] = None
    ground_truth: Optional[dict] = None
