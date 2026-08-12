# pipeline/gemini_analyzer.py
import os
import time
import json
import mimetypes
from typing import Dict, Any, Tuple, List, Optional
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


# Upgraded Prompt Directive aligned with AutoAce AI Trial Specifications
PROMPT_TEXT = (
    "You are an elite acoustic forensics AI specialized in call recording signal analysis.\n"
    "Perform a high-precision audit on this audio clip to achieve top accuracy across all schema fields.\n\n"
    "=== NOISE DETECTION RULES ===\n"
    "1. Isolate the pauses, zero-speech gaps between words, and background layers underneath human voice.\n"
    "2. Inspect for meaningful background sound: office chatter, television, road noise, keyboard clicks, music, mechanical noise, or heavy line hiss/static.\n"
    "3. Set `background_noise_present = true` ONLY if meaningful non-speech sound is audible in the background.\n"
    "4. CRITICAL: Barely perceptible background noise artifacts, micro static, or subtle room atmosphere should NOT automatically count as background noise. In such cases, set `background_noise_present = false` and `background_noise_severity = 'none'`.\n\n"
    "=== FIELD CLASSIFICATION RULES ===\n"
    "- emotional_tone (Enum: 'neutral' | 'satisfied' | 'frustrated' | 'upset' | 'distressed'):\n"
    "  * neutral: calm speech with no strong positive or negative emotion.\n"
    "  * satisfied: pleased, relieved, appreciative, or clearly positive.\n"
    "  * frustrated: annoyed, impatient, or dissatisfied without strong anger or distress.\n"
    "  * upset: clearly angry, agitated, or strongly dissatisfied.\n"
    "  * distressed: highly emotional, overwhelmed, panicked, crying, or escalated.\n"
    "  * RULE: Evaluate pitch inflection and speech cadence, not just volume. Do not infer frustration or distress solely from loudness.\n\n"
    "- emotional_intensity (Enum: 'low' | 'medium' | 'high'):\n"
    "  * MUST be 'low' whenever emotional_tone is 'neutral'. Low = subtle/mild, Medium = clear & sustained, High = strong/escalated.\n\n"
    "- background_noise_present (Boolean):\n"
    "  * `true` if meaningful background noise (chatter, TV, music, road noise, mechanical noise, heavy static) is present.\n"
    "  * `false` if background noise is absent or consists only of barely perceptible artifacts.\n\n"
    "- background_noise_type (String):\n"
    "  * Concise description of dominant noise (e.g. 'office chatter', 'television', 'road noise').\n"
    "  * MUST be empty string '' if background_noise_present is false.\n\n"
    "- background_noise_severity (Enum: 'none' | 'low' | 'medium' | 'high'):\n"
    "  * none: no meaningful noise (background_noise_present = false).\n"
    "  * low: audible but does not interfere with speech.\n"
    "  * medium: occasionally interferes with understanding.\n"
    "  * high: materially impairs conversation or analysis.\n\n"
    "- audio_quality (Enum: 'clear' | 'slightly_impaired' | 'severely_impaired'):\n"
    "  * Overall technical audio quality independent of emotional tone or background noise.\n"
    "  * clear: good overall technical quality.\n"
    "  * slightly_impaired: mild distortion, low volume, minor clipping, or echo.\n"
    "  * severely_impaired: heavy distortion, robotic audio, severe packet loss, or muffled speech.\n\n"
    "- speaker_overlap_present (Boolean): `true` if two or more speakers talk at the same time enough to affect understanding or analysis.\n"
    "- long_silence_present (Boolean): `true` if the clip contains an unusually long period of silence or dead air indicating a call-flow problem.\n"
    "- confidence (Number: 0.0 to 1.0): Model confidence in overall result (1.0 = high, 0.0 = substantial uncertainty)."
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
                temperature=0.1,  # Low temperature eliminates random "clean" noise drops
            ),
        )

        if response.parsed:
            result = response.parsed
        else:
            result = AudioAnalysisResult.model_validate_json(response.text)

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
            "is_batch": False
        }

        return result, usage_stats

    except Exception as e:
        logger.error(f"[{filename}] Gemini Single API call failed: {e}")
        raise e

def process_audio_batch_job(audio_paths: List[str]) -> List[Tuple[str, Optional[AudioAnalysisResult], Dict[str, Any]]]:
    """
    Submits a native Gemini Async Batch Job for Audio Files via JSONL Input File upload.
    Waits for files to transition to ACTIVE state to prevent 400 FAILED_PRECONDITION errors.
    """
    client = get_gemini_client()
    logger.info(f"=== Starting Native Gemini Batch Job for {len(audio_paths)} audio file(s)... ===")

    uploaded_files = []
    file_info_map = {}
    jsonl_lines = []

    try:
        # 1. Upload audio clips & WAIT for them to become ACTIVE
        remote_audio_files = []
        for idx, path in enumerate(audio_paths):
            fname = os.path.basename(path)
            duration_sec = get_audio_duration_seconds(path)
            duration_formatted = format_duration_human(duration_sec)
            
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith("audio/"):
                ext = os.path.splitext(path)[1].lower()
                mime_map = {
                    ".mp3": "audio/mp3",
                    ".ogg": "audio/ogg",
                    ".wav": "audio/wav",
                    ".flac": "audio/flac",
                    ".m4a": "audio/m4a",
                    ".aac": "audio/aac",
                }
                mime_type = mime_map.get(ext, "audio/mp3")

            logger.info(f"[{idx+1}/{len(audio_paths)}] Uploading '{fname}' ({mime_type}) to Gemini Files API...")
            remote_file = client.files.upload(
                file=path, 
                config=types.UploadFileConfig(mime_type=mime_type)
            )
            uploaded_files.append(remote_file)

            # --- CRITICAL FIX: Wait for audio processing to reach ACTIVE state ---
            logger.info(f"Checking processing state for '{fname}' ({remote_file.name})...")
            while remote_file.state and str(remote_file.state).endswith("PROCESSING"):
                time.sleep(2)
                remote_file = client.files.get(name=remote_file.name)
            
            if str(remote_file.state).endswith("FAILED"):
                raise RuntimeError(f"File upload processing failed for {fname}")

            remote_audio_files.append((remote_file, mime_type, fname))

            file_info_map[fname] = {
                "duration_sec": duration_sec,
                "duration_formatted": duration_formatted
            }

        # 2. Construct JSONL Lines
        for idx, (remote_file, mime_type, fname) in enumerate(remote_audio_files):
            key_id = f"req-{idx+1:03d}"
            
            line_obj = {
                "custom_id": key_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "file_data": {
                                        "file_uri": remote_file.uri, 
                                        "mime_type": mime_type
                                    }
                                },
                                {
                                    "text": PROMPT_TEXT
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": AudioAnalysisResult.model_json_schema(),
                        "temperature": 0.1
                    }
                }
            }
            jsonl_lines.append(json.dumps(line_obj))

        # 3. Write JSONL file locally & upload to Files API
        jsonl_path = "temp_batch_requests.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines) + "\n")

        logger.info(f"Uploading batch JSONL payload '{jsonl_path}' to Files API...")
        jsonl_file = client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(display_name="echopulse_audio_batch", mime_type="text/plain")
        )
        uploaded_files.append(jsonl_file)

        # Wait for JSONL file processing if needed
        while jsonl_file.state and str(jsonl_file.state).endswith("PROCESSING"):
            time.sleep(1)
            jsonl_file = client.files.get(name=jsonl_file.name)

        if os.path.exists(jsonl_path):
            os.remove(jsonl_path)

        # 4. Submit Batch Job
        logger.info(f"Submitting Batch Job to Gemini Batches API using source '{jsonl_file.name}'...")
        batch_job = client.batches.create(
            model=MODEL_NAME,
            src=jsonl_file.name,
            config={
                "display_name": "echopulse-audio-batch-job"
            }
        )
        logger.info(f"Batch Job successfully submitted! Job Name: {batch_job.name}, Initial State: {batch_job.state}")

        # 5. Poll until batch job finishes
        poll_interval = 10
        while batch_job.state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            logger.info(f"Batch job status: {batch_job.state}. Polling in {poll_interval}s...")
            time.sleep(poll_interval)
            batch_job = client.batches.get(name=batch_job.name)

        logger.info(f"Batch job finished with final state: {batch_job.state}")

        # 6. Process results
        batch_results = []
        if batch_job.state == "JOB_STATE_SUCCEEDED":
            result_file_name = getattr(batch_job, "output_file", None) or getattr(getattr(batch_job, "dest", None), "file_name", None)
            
            if not result_file_name:
                raise RuntimeError("Batch completed but output file reference was empty.")

            raw_result_bytes = client.files.download(file=result_file_name)
            result_lines = raw_result_bytes.decode('utf-8').splitlines()

            for item_idx, line in enumerate(result_lines):
                if not line.strip():
                    continue
                
                fname = audio_paths[item_idx] if item_idx < len(audio_paths) else f"audio_{item_idx}.wav"
                basename = os.path.basename(fname)
                info = file_info_map.get(basename, {"duration_sec": 0.0, "duration_formatted": "0s"})

                try:
                    resp_obj = json.loads(line)
                    if "response" in resp_obj and resp_obj["response"]:
                        candidates = resp_obj["response"].get("candidates", [])
                        raw_text = candidates[0]["content"]["parts"][0]["text"]
                        parsed_result = AudioAnalysisResult.model_validate_json(raw_text)

                        usage_meta = resp_obj["response"].get("usageMetadata", {})
                        prompt_tokens = usage_meta.get("promptTokenCount", 0)
                        candidate_tokens = usage_meta.get("candidatesTokenCount", 0)
                        total_tokens = usage_meta.get("totalTokenCount", 0)

                        input_cost = (prompt_tokens / 1_000_000) * 0.15
                        output_cost = (candidate_tokens / 1_000_000) * 1.25
                        total_cost_usd = round(input_cost + output_cost, 6)
                        cost_per_min = round((total_cost_usd / (info["duration_sec"] / 60)) if info["duration_sec"] > 0 else 0.0, 6)

                        usage_stats = {
                            "audio_duration_seconds": info["duration_sec"],
                            "audio_duration_formatted": info["duration_formatted"],
                            "prompt_tokens": prompt_tokens,
                            "candidate_tokens": candidate_tokens,
                            "total_tokens": total_tokens,
                            "cost_usd": total_cost_usd,
                            "cost_per_audio_minute_usd": cost_per_min,
                            "is_batch": True
                        }

                        batch_results.append((basename, parsed_result, usage_stats))
                    else:
                        err = resp_obj.get("error", "Unknown batch file error")
                        logger.error(f"File '{basename}' failed in batch execution: {err}")
                        # Log error state without breaking whole batch execution
                        usage_stats = {
                            "audio_duration_seconds": info["duration_sec"],
                            "audio_duration_formatted": info["duration_formatted"],
                            "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
                            "cost_usd": 0.0, "cost_per_audio_minute_usd": 0.0, "is_batch": True
                        }
                        batch_results.append((basename, None, usage_stats))
                except Exception as line_ex:
                    logger.error(f"Failed parsing response line for '{basename}': {line_ex}")
                    usage_stats = {
                        "audio_duration_seconds": info["duration_sec"],
                        "audio_duration_formatted": info["duration_formatted"],
                        "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
                        "cost_usd": 0.0, "cost_per_audio_minute_usd": 0.0, "is_batch": True
                    }
                    batch_results.append((basename, None, usage_stats))
        else:
            raise RuntimeError(f"Gemini Batch Job ended with state: {batch_job.state}")

        return batch_results

    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass
