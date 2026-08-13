# pipeline/runner.py
import os
import time
import json
import pandas as pd
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.schema import AudioAnalysisResult
from pipeline.gemini_analyzer import analyze_audio_with_gemini
from pipeline.audio_utils import format_duration_human
from pipeline.logger import get_logger

logger = get_logger("PipelineRunner")

COST_CEILING_PER_MIN_USD = 0.003  # Target trial constraint
MAX_PARALLEL_WORKERS = 4  # Parallel Gemini API calls (avoids rate limits)


def analyze_audio_file(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Main pipeline entry point for single clip analysis: Runs Multimodal Gemini 3.5 Lite audio analysis.
    """
    filename = os.path.basename(audio_path)
    logger.info(f"=== Starting Multimodal Gemini analysis pipeline for single file '{filename}' ===")

    result, usage_stats = analyze_audio_with_gemini(audio_path)
    
    logger.info(f"=== Successfully completed single analysis for '{filename}' ===")
    return result, usage_stats


def _analyze_single_file_safe(file_path: str, ground_truth_map: Dict) -> Dict[str, Any]:
    """
    Safely analyzes a single audio file. On any error, returns a 'not_processed' result
    instead of raising — ensuring the batch never breaks due to one bad file.
    """
    fname = os.path.basename(file_path)
    try:
        pred, usage = analyze_audio_file(file_path)
        return {
            "filename": fname,
            "status": "success",
            "prediction": pred.model_dump(),
            "usage": usage,
            "ground_truth": ground_truth_map.get(fname)
        }
    except Exception as e:
        logger.error(f"[{fname}] Analysis failed — marking as not_processed: {e}")
        return {
            "filename": fname,
            "status": "not_processed",
            "error": str(e),
            "prediction": None,
            "usage": {
                "audio_duration_seconds": 0.0,
                "audio_duration_formatted": "0s",
                "prompt_tokens": 0,
                "candidate_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_per_audio_minute_usd": 0.0,
                "latency_seconds": 0.0,
                "is_batch": False
            },
            "ground_truth": ground_truth_map.get(fname)
        }


def process_batch(folder_path: str) -> Dict[str, Any]:
    """
    Processes an entire folder of audio clips in PARALLEL using concurrent.futures.
    Each file is analyzed independently via the single-file Gemini API.
    Files that fail are marked 'not_processed' and skipped — they won't break the batch.
    Includes manifest validation (labels.csv), latency tracking, and cost ceiling verification.
    """
    batch_start_time = time.time()
    logger.info(f"--- Starting Parallel Processing for Directory: '{folder_path}' ---")
    
    # Parse manifest (labels.csv) if present
    labels_csv_path = os.path.join(folder_path, "labels.csv")
    manifest_present = False
    ground_truth_map = {}
    manifest_filenames = set()
    
    if os.path.exists(labels_csv_path):
        manifest_present = True
        logger.info(f"Manifest found: Reading ground truth labels from '{labels_csv_path}'")
        try:
            df = pd.read_csv(labels_csv_path)
            for _, row in df.iterrows():
                fname = str(row['name']).strip()
                manifest_filenames.add(fname)
                raw_json = row.get('result_json')
                if pd.notna(raw_json) and str(raw_json).strip():
                    try:
                        ground_truth_map[fname] = json.loads(str(raw_json))
                    except Exception:
                        ground_truth_map[fname] = None
                else:
                    ground_truth_map[fname] = None
            logger.info(f"Parsed {len(manifest_filenames)} manifest rows ({len(ground_truth_map)} labeled) from CSV")
        except Exception as e:
            logger.error(f"Error parsing labels.csv: {e}")

    # Discover audio files
    audio_extensions = {".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac"}
    files = os.listdir(folder_path)
    audio_files = [f for f in sorted(files) if os.path.splitext(f)[1].lower() in audio_extensions]
    logger.info(f"Found {len(audio_files)} audio file(s) to process in parallel (max {MAX_PARALLEL_WORKERS} workers): {audio_files}")

    full_paths = [os.path.join(folder_path, f) for f in audio_files]

    # Manifest matching
    matched_manifest_count = 0
    unmatched_audio_files = []
    for f in audio_files:
        if f in manifest_filenames:
            matched_manifest_count += 1
        else:
            unmatched_audio_files.append(f)

    # Process all files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        future_to_path = {
            executor.submit(_analyze_single_file_safe, path, ground_truth_map): path
            for path in full_paths
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            fname = os.path.basename(path)
            try:
                result = future.result()
                results.append(result)
                status = result["status"]
                logger.info(f"[{fname}] Parallel processing complete — status: {status}")
            except Exception as e:
                # This shouldn't happen since _analyze_single_file_safe catches all errors,
                # but just in case the future itself fails
                logger.error(f"[{fname}] Unexpected parallel executor error: {e}")
                results.append({
                    "filename": fname,
                    "status": "not_processed",
                    "error": str(e),
                    "prediction": None,
                    "usage": {
                        "audio_duration_seconds": 0.0,
                        "audio_duration_formatted": "0s",
                        "prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0,
                        "cost_usd": 0.0, "cost_per_audio_minute_usd": 0.0,
                        "latency_seconds": 0.0, "is_batch": False
                    },
                    "ground_truth": ground_truth_map.get(fname)
                })

    # Sort results by filename to maintain consistent ordering
    results.sort(key=lambda r: r["filename"])

    # Aggregate stats (only from successfully processed files)
    total_audio_duration_seconds = 0.0
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_batch_cost_usd = 0.0

    for item in results:
        usage = item.get("usage") or {}
        total_audio_duration_seconds += usage.get("audio_duration_seconds", 0.0)
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_candidate_tokens += usage.get("candidate_tokens", 0)
        total_batch_cost_usd += usage.get("cost_usd", 0.0)

    total_batch_latency_sec = round(time.time() - batch_start_time, 2)

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "not_processed")
    logger.info(
        f"--- Parallel Processing Complete: {success_count} succeeded, {skipped_count} skipped, "
        f"{len(results)} total in {total_batch_latency_sec}s ---"
    )
    
    cost_per_audio_min = round(
        (total_batch_cost_usd / (total_audio_duration_seconds / 60)) if total_audio_duration_seconds > 0 else 0.0, 6
    )
    is_compliant = cost_per_audio_min <= COST_CEILING_PER_MIN_USD

    return {
        "total_files": len(results),
        "processed_count": success_count,
        "skipped_count": skipped_count,
        "manifest_present": manifest_present,
        "matched_manifest_count": matched_manifest_count,
        "unmatched_audio_files": unmatched_audio_files,
        "total_audio_duration_seconds": round(total_audio_duration_seconds, 2),
        "total_audio_duration_formatted": format_duration_human(total_audio_duration_seconds),
        "total_prompt_tokens": total_prompt_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_tokens": total_prompt_tokens + total_candidate_tokens,
        "total_batch_cost_usd": round(total_batch_cost_usd, 6),
        "cost_per_audio_minute_usd": cost_per_audio_min,
        "cost_ceiling_per_min_usd": COST_CEILING_PER_MIN_USD,
        "cost_ceiling_compliant": is_compliant,
        "total_batch_latency_seconds": total_batch_latency_sec,
        "results": results
    }
