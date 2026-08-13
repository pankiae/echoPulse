# pipeline/gemini_analyzer.py
import os
import time
import json
import mimetypes
from typing import Dict, Any, Tuple, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pipeline.schema import AudioAnalysisResult
from pipeline.logger import get_logger
from pipeline.audio_utils import get_audio_duration_seconds, format_duration_human, compute_acoustic_noise_metrics

# Load environment variables from .env file
load_dotenv(override=True)

logger = get_logger("GeminiAnalyzer")

MODEL_NAME = "gemini-3.5-flash-lite"

_client_cache: Dict[str, genai.Client] = {}


def get_gemini_client() -> genai.Client:
    """
    Dynamically fetches Gemini client evaluating GEMINI_API_KEY environment variable on every call.
    Reuses cached genai.Client instance per API key while warning if key is missing.
    """
    global _client_cache
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    
    if not api_key:
        logger.warning("GEMINI_API_KEY environment variable not set. Initializing client without explicit key.")
    
    if api_key not in _client_cache:
        _client_cache[api_key] = genai.Client(api_key=api_key if api_key else None)
            
    return _client_cache[api_key]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry_error_callback=lambda retry_state: logger.warning(f"Retrying Gemini API call (attempt {retry_state.attempt_number})..."),
    reraise=True
)
def _call_gemini_with_retry(client: genai.Client, audio_part: types.Part, prompt_payload: str):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=[audio_part, prompt_payload],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AudioAnalysisResult,
            temperature=0.1,  # Low temperature eliminates random "clean" noise drops
        ),
    )


# Upgraded Prompt Directive aligned with AutoAce AI Trial Specifications
PROMPT_TEXT = (
"Analyze the audio on the basis of the given instructions. Focus on the background noise!!!"
)


def analyze_audio_with_gemini(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Analyzes a single audio clip synchronously via Google Gemini Flash Multimodal LLM.
    Returns (result_schema, usage_stats_dict).
    """
    start_time = time.time()
    filename = os.path.basename(audio_path)
    logger.info(f"[{filename}] Initializing Gemini 3.5 Lite Single Audio Analysis...")

    audio_duration_sec = get_audio_duration_seconds(audio_path)
    audio_duration_formatted = format_duration_human(audio_duration_sec)
    acoustic_metrics = compute_acoustic_noise_metrics(audio_path)

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

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    
    prompt_payload = f"{PROMPT_TEXT}\n\n=== ACOUSTIC PRE-ANALYSIS ===\n{acoustic_metrics.get('acoustic_hint', '')}"

    try:
        response = _call_gemini_with_retry(client, audio_part, prompt_payload)

        if response.parsed:
            result = response.parsed
        else:
            result = AudioAnalysisResult.model_validate_json(response.text)

        latency_sec = round(time.time() - start_time, 2)
        cost_usd = 0.0
        prompt_tokens = 0
        candidate_tokens = 0
        total_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            candidate_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0
            
            # Standard Tier: $0.30 / 1M input, $2.50 / 1M output
            input_cost = (prompt_tokens / 1_000_000) * 0.30
            output_cost = (candidate_tokens / 1_000_000) * 2.50
            cost_usd = round(input_cost + output_cost, 6)

        cost_per_min = round((cost_usd / (audio_duration_sec / 60)) if audio_duration_sec > 0 else 0.0, 6)

        usage_stats = {
            "audio_duration_seconds": audio_duration_sec,
            "audio_duration_formatted": audio_duration_formatted,
            "prompt_tokens": prompt_tokens,
            "candidate_tokens": candidate_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "cost_per_audio_minute_usd": cost_per_min,
            "latency_seconds": latency_sec,
            "is_batch": False
        }

        return result, usage_stats

    except Exception as e:
        logger.error(f"[{filename}] Gemini Single API call failed: {e}")
        raise e

