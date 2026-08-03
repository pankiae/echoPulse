# pipeline/gemini_analyzer.py
import os
import mimetypes
from typing import Dict, Any, Tuple
from google import genai
from google.genai import types
from pipeline.schema import AudioAnalysisResult
from pipeline.logger import get_logger
from pipeline.audio_utils import get_audio_duration_seconds, format_duration_human

logger = get_logger("GeminiAnalyzer")

MODEL_NAME = "gemini-3.5-flash-lite"

_client_cache: Dict[str, genai.Client] = {}


def get_gemini_client() -> genai.Client:
    """
    Dynamically fetches Gemini client evaluating GEMINI_API_KEY environment variable on every call.
    Reuses cached genai.Client instance per API key while warning if key is missing.
    """
    global _client_cache
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Initializing client without explicit key.")
    
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(api_key=api_key if api_key else None)
            
    return _client_cache[api_key]


def analyze_audio_with_gemini(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Analyzes an audio clip directly using Google Gemini Flash Multimodal LLM
    with native Structured Output (Pydantic schema adhering strictly to AutoAce specifications).
    Calculates audio duration, token usage, and cost.
    Returns (result_schema, usage_stats_dict).
    """
    filename = os.path.basename(audio_path)
    logger.info(f"[{filename}] Initializing Gemini 3.5 Lite Multimodal Audio Analysis (P95 Noise Calibration)...")

    audio_duration_sec = get_audio_duration_seconds(audio_path)
    audio_duration_formatted = format_duration_human(audio_duration_sec)

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

    logger.info(f"[{filename}] Reading audio file bytes ({mime_type}, duration={audio_duration_formatted})...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    prompt = (
        "You are an elite acoustic forensics AI specialized in call recording signal analysis. "
        "Perform a high-precision multi-pass audit on this audio clip to achieve P95 accuracy across all 9 schema fields.\n\n"
        "=== ACOUSTIC NOISE AUDIT PROTOCOL (CRITICAL FOR P95 ACCURACY) ===\n"
        "1. STEP-BY-STEP NOISE DETECTION:\n"
        "   - Listen to pauses, gaps between words, and the silent background layer underneath human voice.\n"
        "   - Check for CONSTANT background noise: electrical hiss, line static, hum, fan noise, air conditioning, white noise.\n"
        "   - Check for INTERMITTENT background noise: TV speech/chatter, background television sound, office background voices, road/vehicle noise, keyboard clicks, mic rustle.\n"
        "   - CRITICAL RULE: Human speech clarity DOES NOT cancel background noise! If speech is clear BUT background static or TV is audible, set `background_noise_present = true` and `audio_quality = 'clear'` (or `'slightly_impaired'`). Noise and quality are SEPARATE metrics.\n"
        "   - Set `background_noise_present = false` ONLY if the background is completely clean or studio silent.\n\n"
        "2. FIELD DEFINITIONS & RULES:\n"
        "   - emotional_tone (Enum: 'neutral' | 'satisfied' | 'frustrated' | 'upset' | 'distressed'):\n"
        "     * neutral: calm speech with no strong positive or negative emotional polarity.\n"
        "     * satisfied: pleased, relieved, appreciative, or warm tone.\n"
        "     * frustrated: annoyed, impatient, passive-aggressive, or dissatisfied.\n"
        "     * upset: angry, agitated, shouting, or direct confrontation.\n"
        "     * distressed: overwhelmed, crying, panicked, or in severe distress.\n"
        "     * RULE: Do NOT infer frustration/distress from volume alone. Evaluate pitch inflection and speech rhythm.\n\n"
        "   - emotional_intensity (Enum: 'low' | 'medium' | 'high'):\n"
        "     * low: subtle/mild. MUST be 'low' whenever emotional_tone is 'neutral'.\n"
        "     * medium: clear and sustained emotion.\n"
        "     * high: strong, escalated, or intense.\n\n"
        "   - background_noise_present (Boolean: true | false):\n"
        "     * `true` if ANY audible background sound, static, TV, chatter, hum, music, or environmental noise exists.\n"
        "     * `false` ONLY if the background soundscape has zero non-speech noise.\n\n"
        "   - background_noise_type (String):\n"
        "     * Specific description: e.g. 'sharp static', 'television', 'office chatter', 'hiss', 'road noise', 'music', 'wind', 'keyboard typing'.\n"
        "     * MUST be empty string '' if background_noise_present is false.\n\n"
        "   - background_noise_severity (Enum: 'none' | 'low' | 'medium' | 'high'):\n"
        "     * none: zero noise (background_noise_present = false).\n"
        "     * low: audible background noise that does not interfere with speech.\n"
        "     * medium: clearly audible noise that occasionally competes with understanding.\n"
        "     * high: loud static, loud TV, or dominating background noise.\n\n"
        "   - audio_quality (Enum: 'clear' | 'slightly_impaired' | 'severely_impaired'):\n"
        "     * Technical quality independent of emotion or background noise presence.\n\n"
        "   - speaker_overlap_present (Boolean: true | false):\n"
        "     * `true` if multiple voices talk simultaneously.\n\n"
        "   - long_silence_present (Boolean: true | false):\n"
        "     * `true` if there is continuous dead air / silence (>4 seconds).\n\n"
        "   - confidence (Number: 0.0 to 1.0):\n"
        "     * Confidence score from 0.0 to 1.0 based on signal clarity."
    )

    logger.info(f"[{filename}] Sending prompt and audio tensor to Gemini '{MODEL_NAME}' with structured Pydantic response_schema...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioAnalysisResult,
                temperature=0.0,
            ),
        )

        logger.info(f"[{filename}] Gemini response received successfully.")
        
        if response.parsed:
            result = response.parsed
        else:
            result = AudioAnalysisResult.model_validate_json(response.text)

        usage_stats = {
            "audio_duration_seconds": audio_duration_sec,
            "audio_duration_formatted": audio_duration_formatted,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }

        # Calculate exact token consumption and cost
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            candidate_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0
            
            # Pricing for gemini-3.5-flash-lite Standard Paid Tier ($0.30 / 1M input, $2.50 / 1M output)
            input_cost = (prompt_tokens / 1_000_000) * 0.30
            output_cost = (candidate_tokens / 1_000_000) * 2.50
            total_cost_usd = input_cost + output_cost

            usage_stats = {
                "audio_duration_seconds": audio_duration_sec,
                "audio_duration_formatted": audio_duration_formatted,
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(total_cost_usd, 6)
            }
            
            logger.info(
                f"[{filename}] Audio Duration: {audio_duration_formatted} ({audio_duration_sec}s) | "
                f"Tokens: Prompt={prompt_tokens}, Output={candidate_tokens}, Total={total_tokens} | "
                f"Cost: ${total_cost_usd:.6f} USD (~{total_cost_usd*100:.4f}¢)"
            )

        logger.info(f"[{filename}] Analysis complete: tone='{result.emotional_tone}', intensity='{result.emotional_intensity}', noise_present={result.background_noise_present}, noise_type='{result.background_noise_type}', confidence={result.confidence}")
        return result, usage_stats

    except Exception as e:
        logger.error(f"[{filename}] Gemini API call failed: {e}")
        raise e
