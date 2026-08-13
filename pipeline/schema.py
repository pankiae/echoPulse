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
            "Primary emotional state of speaker evaluated from pitch inflection, vocal tone, and speech cadence. "
            "'neutral': calm, professional speech with no clear positive or negative emotion. "
            "'satisfied': pleased, relieved, appreciative, or clearly positive. "
            "'frustrated': annoyed, impatient, or dissatisfied without strong anger or distress. "
            "'upset': clearly angry, agitated, or strongly dissatisfied. "
            "'distressed': highly emotional, overwhelmed, panicked, crying, or emotionally escalated."
        ),
    )
    emotional_intensity: EmotionalIntensity = Field(
        ...,
        description=(
            "Strength or degree of expressed emotion. "
            "'low': subtle or mild (MUST be 'low' if emotional_tone is 'neutral'). "
            "'medium': clear, noticeable, and sustained emotion. "
            "'high': intense, escalated, or strong emotion."
        ),
    )
    background_noise_present: bool = Field(
        ...,
        description=(
            "Set true if any background sound, ambient room atmosphere, secondary human chatter, television, "
            "traffic, fan/HVAC, keyboard typing, or transmission static is audible during speech or pauses. "
            "Set false ONLY if the background layer is completely silent and clean."
        ),
    )
    background_noise_type: str = Field(
        default="",
        description=(
            "Specific label describing the dominant acoustic background noise source. "
            "Prioritize specific sources if audible: 'office chatter', 'background voices', 'television', 'radio broadcast', "
            "'road noise', 'traffic', 'keyboard typing', 'mechanical noise', 'fan noise', 'air conditioning', 'music'. "
            "Use 'line hiss' or 'radio static' ONLY if electrical noise exists without voices/TV/traffic. "
            "Must be empty string '' if background_noise_present is false."
        ),
    )
    background_noise_severity: BackgroundNoiseSeverity = Field(
        ...,
        description=(
            "Audible severity level of background noise. "
            "'none': no background noise present (background_noise_present = false). "
            "'low': audible background noise or static that does not interfere with comprehension. "
            "'medium': background noise that occasionally interferes with speech comprehension. "
            "'high': heavy background noise/static that severely impairs conversation or analysis."
        ),
    )
    audio_quality: AudioQuality = Field(
        ...,
        description=(
            "Overall technical signal quality of recording. "
            "'clear': good technical audio quality. "
            "'slightly_impaired': mild distortion, low volume, minor clipping, or echo. "
            "'severely_impaired': severe distortion, muffled speech, robotic audio, or heavy packet loss."
        ),
    )
    speaker_overlap_present: bool = Field(
        ...,
        description=(
            "Set true if two or more distinct human voices speak simultaneously, talk over each other, "
            "or interrupt one another at any point in the recording. Set false if speakers take turns cleanly."
        ),
    )
    long_silence_present: bool = Field(
        ...,
        description=(
            "Set true if the clip contains extended dead air, uncomfortably long silence (> 2.5 seconds), "
            "or audio dropout between speech turns. Set false for standard conversational pauses."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence score for overall audio evaluation from 0.0 (substantial uncertainty) to 1.0 (high certainty)."
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
