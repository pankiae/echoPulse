# pipeline/gemini_analyzer.py
import os
import mimetypes
from google import genai
from google.genai import types
from pipeline.schema import AudioAnalysisResult
from pipeline.logger import get_logger

logger = get_logger("GeminiAnalyzer")

# Recommended multimodal Gemini 3.5 / Flash model for structured audio extraction
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

_client = None


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Initializing client without explicit key.")
        _client = genai.Client(api_key=api_key)
    return _client


def analyze_audio_with_gemini(audio_path: str) -> AudioAnalysisResult:
    """
    Analyzes an audio clip directly using Google Gemini Flash Multimodal LLM
    with native Structured Output (Pydantic schema).
    """
    filename = os.path.basename(audio_path)
    logger.info(f"[{filename}] Initializing Gemini 3.5 Lite Multimodal Audio Analysis...")

    client = get_gemini_client()

    # Determine mime-type based on extension
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
        "You are an expert audio analytics and voice segmentation AI. "
        "Analyze this audio clip thoroughly and output a strictly formatted structured JSON response matching the required schema.\n\n"
        "Instructions:\n"
        "1. emotional_tone: Primary emotional tone expressed by customer ('neutral', 'satisfied', 'frustrated', 'upset', 'distressed').\n"
        "2. emotional_intensity: Strength of the emotional tone ('low', 'medium', 'high'). If tone is 'neutral', intensity must be 'low'.\n"
        "3. background_noise_present: true if non-speech background sound is audible, false otherwise.\n"
        "4. background_noise_type: Description of dominant background noise (e.g., 'office chatter', 'TV', 'sharp static', 'traffic', 'siren'). Empty string '' if no background noise.\n"
        "5. background_noise_severity: Noise impact on clarity ('none', 'low', 'medium', 'high'). Must be 'none' if background_noise_present is false.\n"
        "6. audio_quality: Technical audio quality ('clear', 'slightly_impaired', 'severely_impaired').\n"
        "7. speaker_overlap_present: true if multiple speakers talk at the same time.\n"
        "8. long_silence_present: true if there is an unusual period of continuous dead air/silence (> 4 seconds).\n"
        "9. confidence: Float score between 0.0 and 1.0 representing analysis confidence."
    )

    logger.info(f"[{filename}] Sending multimodal prompt to Gemini model '{MODEL_NAME}' with structured Pydantic response_schema...")

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

        logger.info(f"[{filename}] Gemini raw output response received successfully.")
        
        # Parsed output is available in response.parsed or via standard validation
        if response.parsed:
            result = response.parsed
        else:
            result = AudioAnalysisResult.model_validate_json(response.text)

        logger.info(f"[{filename}] Analysis complete: tone='{result.emotional_tone}', noise='{result.background_noise_type}', confidence={result.confidence}")
        return result

    except Exception as e:
        logger.error(f"[{filename}] Gemini API call failed: {e}")
        # Graceful fallback in case API key is missing or quota error occurs
        raise e
