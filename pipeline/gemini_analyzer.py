# pipeline/gemini_analyzer.py
import os
import time
import mimetypes
from typing import Dict, Any, Tuple, List
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


PROMPT_TEXT = (
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


def analyze_audio_with_gemini(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Analyzes a single audio clip synchronously via Google Gemini Flash Multimodal LLM.
    Returns (result_schema, usage_stats_dict).
    """
    filename = os.path.basename(audio_path)
    logger.info(f"[{filename}] Initializing Gemini 3.5 Lite Single Audio Analysis...")

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

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[audio_part, PROMPT_TEXT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AudioAnalysisResult,
                temperature=0.0,
            ),
        )

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
            "cost_usd": 0.0,
            "is_batch": False
        }

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            candidate_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0
            
            # Standard Tier: $0.30 / 1M input, $2.50 / 1M output
            input_cost = (prompt_tokens / 1_000_000) * 0.30
            output_cost = (candidate_tokens / 1_000_000) * 2.50
            total_cost_usd = input_cost + output_cost

            usage_stats = {
                "audio_duration_seconds": audio_duration_sec,
                "audio_duration_formatted": audio_duration_formatted,
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(total_cost_usd, 6),
                "is_batch": False
            }

        return result, usage_stats

    except Exception as e:
        logger.error(f"[{filename}] Gemini Single API call failed: {e}")
        raise e


def process_audio_batch_job(audio_paths: List[str]) -> List[Tuple[str, AudioAnalysisResult, Dict[str, Any]]]:
    """
    Submits a native Gemini Async Batch Job via client.batches.create API with 50% discount rate ($0.15/1M in, $1.25/1M out).
    Uploads audio files via client.files.upload(), submits batch job, polls until completion, and validates Pydantic output.
    """
    client = get_gemini_client()
    logger.info(f"=== Starting Native Gemini Batch Job for {len(audio_paths)} audio file(s)... ===")

    # 1. Upload files and build batch requests list
    batch_requests = []
    uploaded_files = []
    file_info_map = {}

    try:
        for idx, path in enumerate(audio_paths):
            fname = os.path.basename(path)
            duration_sec = get_audio_duration_seconds(path)
            duration_formatted = format_duration_human(duration_sec)
            
            logger.info(f"[{idx+1}/{len(audio_paths)}] Uploading '{fname}' to Gemini Files API...")
            remote_file = client.files.upload(file=path)
            uploaded_files.append(remote_file)

            file_info_map[fname] = {
                "duration_sec": duration_sec,
                "duration_formatted": duration_formatted
            }

            custom_id = f"audio-{idx+1:03d}-{fname}"
            batch_requests.append({
                "custom_id": custom_id,
                "request": types.GenerateContentConfig(
                    model=MODEL_NAME,
                    contents=[remote_file, PROMPT_TEXT],
                    response_mime_type="application/json",
                    response_schema=AudioAnalysisResult.model_json_schema(),
                    temperature=0.0,
                )
            })

        # 2. Submit Batch Job
        logger.info(f"Submitting Batch Job to Gemini Batches API for {len(batch_requests)} item(s)...")
        batch_job = client.batches.create(
            model=MODEL_NAME,
            requests=batch_requests
        )
        logger.info(f"Batch Job successfully submitted! Job Name/ID: {batch_job.name}, Initial State: {batch_job.state}")

        # 3. Poll Batch Job status
        poll_interval = 5
        while batch_job.state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            logger.info(f"Batch job '{batch_job.name}' in progress (state={batch_job.state}). Polling in {poll_interval}s...")
            time.sleep(poll_interval)
            batch_job = client.batches.get(name=batch_job.name)

        logger.info(f"Batch job '{batch_job.name}' finished with final state: {batch_job.state}")

        batch_results = []
        if batch_job.state == "JOB_STATE_SUCCEEDED":
            for item_idx, res in enumerate(batch_job.results):
                fname = audio_paths[item_idx]
                basename = os.path.basename(fname)
                info = file_info_map.get(basename, {"duration_sec": 0.0, "duration_formatted": "0s"})
                
                if res.response:
                    parsed_result = AudioAnalysisResult.model_validate_json(res.response.text)
                    
                    # Compute batch usage & 50% discounted pricing ($0.15/1M input, $1.25/1M output)
                    prompt_tokens = 0
                    candidate_tokens = 0
                    total_tokens = 0
                    total_cost_usd = 0.0

                    if hasattr(res.response, "usage_metadata") and res.response.usage_metadata:
                        usage = res.response.usage_metadata
                        prompt_tokens = usage.prompt_token_count or 0
                        candidate_tokens = usage.candidates_token_count or 0
                        total_tokens = usage.total_token_count or 0
                        
                        input_cost = (prompt_tokens / 1_000_000) * 0.15
                        output_cost = (candidate_tokens / 1_000_000) * 1.25
                        total_cost_usd = input_cost + output_cost

                    usage_stats = {
                        "audio_duration_seconds": info["duration_sec"],
                        "audio_duration_formatted": info["duration_formatted"],
                        "prompt_tokens": prompt_tokens,
                        "candidate_tokens": candidate_tokens,
                        "total_tokens": total_tokens,
                        "cost_usd": round(total_cost_usd, 6),
                        "is_batch": True
                    }

                    batch_results.append((basename, parsed_result, usage_stats))
                else:
                    raise RuntimeError(f"Batch item '{res.custom_id}' failed in batch response")
        else:
            raise RuntimeError(f"Gemini Batch Job failed with state '{batch_job.state}'")

        return batch_results

    finally:
        # Clean up uploaded temporary files from Gemini Files API
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass
