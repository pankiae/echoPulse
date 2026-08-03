# pipeline/gemini_analyzer.py
import os
import mimetypes
from google import genai
from google.genai import types
from pipeline.schema import AudioAnalysisResult
from pipeline.logger import get_logger

logger = get_logger("GeminiAnalyzer")

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def get_gemini_client() -> genai.Client:
    """
    Dynamically fetches Gemini client evaluating GEMINI_API_KEY environment variable on every call.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Initializing client without explicit key.")
    return genai.Client(api_key=api_key)


def analyze_audio_with_gemini(audio_path: str) -> AudioAnalysisResult:
    """
    Analyzes an audio clip directly using Google Gemini Flash Multimodal LLM
    with native Structured Output (Pydantic schema adhering strictly to AutoAce specifications).
    """
    filename = os.path.basename(audio_path)
    logger.info(f"[{filename}] Initializing Gemini 3.5 Lite Multimodal Audio Analysis...")

    client = get_gemini_client()

    mime_type, _ = mimetypes.guess_type(audio_path)
    if not mime_type or not mime_type.startswith("audio/"):
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {
            ".mp3": "audio/mp3",
            ".ogg": "audio/ogg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/m4a",
            ".aac": "audio/aac",
        }
        mime_type = mime_map.get(ext, "audio/mp3")

    logger.info(f"[{filename}] Reading audio file bytes ({mime_type})...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    prompt = (
        "You are an expert audio analytics and voice segmentation AI for call recording evaluation. "
        "Analyze this audio clip thoroughly and output a strictly formatted structured JSON response matching the required schema.\n\n"
        "EVALUATION RULES & DEFINITIONS:\n"
        "1. emotional_tone (Enum: 'neutral' | 'satisfied' | 'frustrated' | 'upset' | 'distressed'):\n"
        "   - neutral: no clear positive or negative emotion.\n"
        "   - satisfied: pleased, relieved, appreciative, or clearly positive.\n"
        "   - frustrated: annoyed, impatient, or dissatisfied without strong anger or distress.\n"
        "   - upset: clearly angry, agitated, or strongly dissatisfied.\n"
        "   - distressed: highly emotional, overwhelmed, panicked, crying, or emotionally escalated.\n"
        "   *Note: Do NOT infer frustration or distress solely from loudness.\n\n"
        "2. emotional_intensity (Enum: 'low' | 'medium' | 'high'):\n"
        "   - low: subtle or mild.\n"
        "   - medium: clear and sustained.\n"
        "   - high: strong, escalated, or likely to require attention.\n\n"
        "3. background_noise_present (Boolean: true | false):\n"
        "   - true if meaningful non-speech sound is audible. Barely perceptible artifacts should NOT automatically count.\n\n"
        "4. background_noise_type (String):\n"
        "   - A concise description of dominant background noise, such as 'office chatter', 'music', 'road noise', 'television', 'keyboard typing', 'wind', 'sharp static', or 'mechanical noise'. Empty string '' if background_noise_present is false.\n\n"
        "5. background_noise_severity (Enum: 'none' | 'low' | 'medium' | 'high'):\n"
        "   - none: no meaningful noise.\n"
        "   - low: audible but does not interfere.\n"
        "   - medium: occasionally interferes with understanding.\n"
        "   - high: materially impairs conversation or analysis.\n\n"
        "6. audio_quality (Enum: 'clear' | 'slightly_impaired' | 'severely_impaired'):\n"
        "   - Technical quality independent of emotion. Consider distortion, clipping, echo, static, low volume, muffled speech, robotic audio, and packet loss.\n"
        "   *Note: Do NOT infer background noise solely from poor audio quality.\n\n"
        "7. speaker_overlap_present (Boolean: true | false):\n"
        "   - true if two or more speakers talk at the same time enough to affect understanding or analysis.\n\n"
        "8. long_silence_present (Boolean: true | false):\n"
        "   - true if the clip contains an unusually long period of silence or dead air that may indicate a call-flow or audio problem.\n\n"
        "9. confidence (Number: 0.0 to 1.0):\n"
        "   - The model's confidence in the overall result. Values near 1.0 indicate high confidence; values near 0.0 indicate substantial uncertainty."
    )

    logger.info(f"[{filename}] Sending prompt and audio tensor to Gemini '{MODEL_NAME}' with structured Pydantic response_schema...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioAnalysisResult,
                temperature=0.1,
            ),
        )

        logger.info(f"[{filename}] Gemini response received successfully.")
        
        if response.parsed:
            result = response.parsed
        else:
            result = AudioAnalysisResult.model_validate_json(response.text)

        logger.info(f"[{filename}] Analysis complete: tone='{result.emotional_tone}', intensity='{result.emotional_intensity}', noise='{result.background_noise_type}', confidence={result.confidence}")
        return result

    except Exception as e:
        logger.error(f"[{filename}] Gemini API call failed: {e}")
        raise e
