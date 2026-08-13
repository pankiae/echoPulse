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
    "You are an elite acoustic forensics AI specialized in call recording signal analysis.\n"
    "Perform a high-precision audit on this audio clip to achieve top accuracy across all schema fields.\n\n"
    "=== BACKGROUND NOISE DETECTION & IDENTIFICATION RULES ===\n"
    "1. Systematically listen to the pauses between words, zero-speech gaps, and continuous audio layers beneath the human voice.\n"
    "2. Set `background_noise_present = true` whenever ANY non-speech background sound, room atmosphere, vehicle sound, secondary voices, or line noise is audible.\n"
    "3. Set `background_noise_present = false` ONLY if the background is completely silent and clean without any audible background noise.\n\n"
    "=== BACKGROUND NOISE SOURCE CLASSIFICATION HIERARCHY ===\n"
    "When `background_noise_present = true`, classify `background_noise_type` according to the DOMINANT audible source:\n"
    "- If secondary human voices, ambient talking, or call center background chatter is audible -> MUST classify as 'office chatter' or 'background voices'.\n"
    "- If audio/speech from a TV show, broadcast, or screen is audible -> MUST classify as 'television' or 'radio broadcast'.\n"
    "- If car engines, tire rolling, traffic, or outdoor vehicle sound is audible -> MUST classify as 'road noise' or 'traffic'.\n"
    "- If key clicks, typing, or office equipment sound is audible -> MUST classify as 'keyboard typing' or 'mechanical noise'.\n"
    "- If HVAC, fan blowing, or air flow is audible -> MUST classify as 'fan noise' or 'air conditioning'.\n"
    "- If background music or instrumental tunes are playing -> MUST classify as 'music'.\n"
    "- ONLY if the background is electrical RF static or line hiss with NO chatter, TV, traffic, typing, or mechanical sounds -> classify as 'line hiss' or 'radio static'.\n"
    "- When `background_noise_present = false`, MUST set `background_noise_type = ''` and `background_noise_severity = 'none'`.\n\n"
    "=== FIELD CLASSIFICATION RULES ===\n"
    "- emotional_tone (Enum: 'neutral' | 'satisfied' | 'frustrated' | 'upset' | 'distressed'):\n"
    "  * neutral: calm speech with no clear positive or negative emotion.\n"
    "  * satisfied: pleased, relieved, appreciative, or clearly positive.\n"
    "  * frustrated: annoyed, impatient, or dissatisfied without strong anger or distress.\n"
    "  * upset: clearly angry, agitated, or strongly dissatisfied.\n"
    "  * distressed: highly emotional, overwhelmed, panicked, crying, or escalated.\n"
    "  * RULE: Evaluate pitch inflection and speech cadence, not just volume. Do not infer frustration or distress solely from loudness.\n\n"
    "- emotional_intensity (Enum: 'low' | 'medium' | 'high'):\n"
    "  * MUST be 'low' whenever emotional_tone is 'neutral'. Low = subtle/mild, Medium = clear & sustained, High = strong/escalated.\n\n"
    "- background_noise_severity (Enum: 'none' | 'low' | 'medium' | 'high'):\n"
    "  * none: no background noise (background_noise_present = false).\n"
    "  * low: audible background noise/static that does not interfere with understanding.\n"
    "  * medium: background noise/static that occasionally interferes with understanding.\n"
    "  * high: severe noise/static that materially impairs conversation or analysis.\n\n"
    "- audio_quality (Enum: 'clear' | 'slightly_impaired' | 'severely_impaired'):\n"
    "  * Overall technical audio quality independent of emotional tone.\n"
    "  * clear: good overall technical quality.\n"
    "  * slightly_impaired: mild distortion, low volume, minor clipping, or echo.\n"
    "  * severely_impaired: heavy distortion, robotic audio, severe packet loss, or muffled speech.\n\n"
    "=== SPEAKER OVERLAP DETECTION RULES ===\n"
    "1. Listen systematically for simultaneous speech, crosstalk, interruptions, or two people talking concurrently.\n"
    "2. Set `speaker_overlap_present = true` whenever two or more distinct speakers talk at the exact same time, interrupt each other, or speak simultaneously.\n"
    "3. Set `speaker_overlap_present = false` ONLY if speakers take turns cleanly without talking over one another.\n\n"
    "=== LONG SILENCE DETECTION RULES ===\n"
    "1. Inspect for dead air, uncomfortably long gaps (> 2.5 seconds of silence), or extended conversational pauses.\n"
    "2. Set `long_silence_present = true` if the clip contains unusually long period of silence or dead air indicating a call-flow problem or audio dropout.\n"
    "3. Set `long_silence_present = false` if speech flow is natural without prolonged dead air.\n\n"
    "- confidence (Number: 0.0 to 1.0): Model confidence in overall result (1.0 = high, 0.0 = substantial uncertainty)."
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


def process_audio_batch_job(audio_paths: List[str]) -> List[Tuple[str, Optional[AudioAnalysisResult], Dict[str, Any]]]:
    """
    Submits a native Gemini Async Batch Job for Audio Files via JSONL Input File upload.
    Waits for files to transition to ACTIVE state to prevent 400 FAILED_PRECONDITION errors.
    """
    batch_start_time = time.time()
    client = get_gemini_client()
    logger.info(f"=== Starting Native Gemini Batch Job for {len(audio_paths)} audio file(s)... ===")

    uploaded_files = []
    file_info_map = {}
    req_id_to_file_map = {}
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

            req_id = f"req-{idx+1:03d}"
            remote_audio_files.append((remote_file, mime_type, fname, req_id, path))
            req_id_to_file_map[req_id] = fname

            file_info_map[fname] = {
                "duration_sec": duration_sec,
                "duration_formatted": duration_formatted
            }

        # 2. Construct JSONL Lines
        for remote_file, mime_type, fname, req_id, local_path in remote_audio_files:
            ac_metrics = compute_acoustic_noise_metrics(local_path)
            prompt_payload = f"{PROMPT_TEXT}\n\n=== ACOUSTIC PRE-ANALYSIS ===\n{ac_metrics.get('acoustic_hint', '')}"

            line_obj = {
                "custom_id": req_id,
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
                                    "text": prompt_payload
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

        batch_total_time = round(time.time() - batch_start_time, 2)
        logger.info(f"Batch job finished with final state: {batch_job.state} in {batch_total_time}s")

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

                try:
                    resp_obj = json.loads(line)
                    custom_id = resp_obj.get("custom_id", f"req-{item_idx+1:03d}")
                    fname = req_id_to_file_map.get(custom_id, audio_paths[item_idx] if item_idx < len(audio_paths) else f"audio_{item_idx}.wav")
                    basename = os.path.basename(fname)
                    info = file_info_map.get(basename, {"duration_sec": 0.0, "duration_formatted": "0s"})

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
                            "latency_seconds": round(batch_total_time / len(audio_paths), 2) if audio_paths else 0.0,
                            "is_batch": True
                        }

                        batch_results.append((basename, parsed_result, usage_stats))
                    else:
                        err = resp_obj.get("error", "Unknown batch file error")
                        logger.error(f"File '{basename}' failed in batch execution: {err}")
                        usage_stats = {
                            "audio_duration_seconds": info["duration_sec"],
                            "audio_duration_formatted": info["duration_formatted"],
                            "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
                            "cost_usd": 0.0, "cost_per_audio_minute_usd": 0.0,
                            "latency_seconds": 0.0, "is_batch": True
                        }
                        batch_results.append((basename, None, usage_stats))
                except Exception as line_ex:
                    logger.error(f"Failed parsing response line: {line_ex}")
                    usage_stats = {
                        "audio_duration_seconds": 0.0, "audio_duration_formatted": "0s",
                        "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
                        "cost_usd": 0.0, "cost_per_audio_minute_usd": 0.0,
                        "latency_seconds": 0.0, "is_batch": True
                    }
                    batch_results.append((f"error_{item_idx}.wav", None, usage_stats))
        else:
            raise RuntimeError(f"Gemini Batch Job ended with state: {batch_job.state}")

        return batch_results

    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass
