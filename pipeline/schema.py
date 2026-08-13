# pipeline/schema.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


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
    emotional_tone: EmotionalTone = Field(
        ...,
        description=(
            "Primary emotional tone expressed by customer. "
            "neutral: no clear positive or negative emotion. "
            "satisfied: pleased, relieved, appreciative, or clearly positive. "
            "frustrated: annoyed, impatient, or dissatisfied without strong anger. "
            "upset: clearly angry, agitated, or strongly dissatisfied. "
            "distressed: highly emotional, overwhelmed, panicked, crying, or emotionally escalated."
        ),
    )
    emotional_intensity: EmotionalIntensity = Field(
        ...,
        description=(
            "Strength of detected emotional tone. "
            "low: subtle or mild. "
            "medium: clear and sustained. "
            "high: strong, escalated, or requiring immediate attention."
        ),
    )
    background_noise_present: bool = Field(
        ...,
        description=(
            "Whether meaningful non-speech sound is audible in the background. "
            "Barely perceptible artifacts should not count as background noise."
        ),
    )
    background_noise_type: str = Field(
        default="",
        description=(
            "A concise description of the dominant background noise, such as office chatter, "
            "music, road noise, television, keyboard typing, wind, or mechanical noise. "
            "Must be empty string '' if background_noise_present is false."
        ),
    )
    background_noise_severity: BackgroundNoiseSeverity = Field(
        ...,
        description=(
            "How much the noise affects the call. "
            "none: no meaningful noise. "
            "low: audible but does not interfere. "
            "medium: occasionally interferes with understanding. "
            "high: materially impairs conversation or analysis."
        ),
    )
    audio_quality: AudioQuality = Field(
        ...,
        description=(
            "Overall technical quality of audio independent of emotion. "
            "clear: good technical quality. "
            "slightly_impaired: mild distortion/clipping/static. "
            "severely_impaired: severe distortion, low volume, muffled speech, robotic audio, or severe packet loss."
        ),
    )
    speaker_overlap_present: bool = Field(
        ...,
        description=(
            "Whether two or more speakers talk at the same time enough to affect understanding or analysis."
        ),
    )
    long_silence_present: bool = Field(
        ...,
        description=(
            "Whether the clip contains an unusually long period of silence or dead air that may indicate a call-flow or audio problem."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "The model's confidence in the overall result from 0.0 (substantial uncertainty) to 1.0 (high confidence)."
        ),
    )

    @model_validator(mode="after")
    def validate_trial_rules(self) -> "AudioAnalysisResult":
        # Rule 1: If background noise is absent, enforce empty type and severity=none
        if not self.background_noise_present:
            self.background_noise_type = ""
            self.background_noise_severity = BackgroundNoiseSeverity.NONE
        elif self.background_noise_severity == BackgroundNoiseSeverity.NONE:
            # If severity was marked as none, set noise_present to False
            self.background_noise_present = False
            self.background_noise_type = ""

        # Rule 2: Neutral tone MUST have low emotional intensity
        if self.emotional_tone == EmotionalTone.NEUTRAL:
            self.emotional_intensity = EmotionalIntensity.LOW

        return self



class BatchFileResult(BaseModel):
    filename: str
    status: str = "success"
    error: Optional[str] = None
    prediction: Optional[AudioAnalysisResult] = None
    ground_truth: Optional[dict] = None
