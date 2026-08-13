# pipeline/runner.py
import os
import time
import json
import pandas as pd
from typing import Dict, Any, Tuple
from pipeline.schema import AudioAnalysisResult
from pipeline.gemini_analyzer import analyze_audio_with_gemini, process_audio_batch_job
from pipeline.audio_utils import format_duration_human
from pipeline.logger import get_logger

logger = get_logger("PipelineRunner")

COST_CEILING_PER_MIN_USD = 0.003  # Target trial constraint


def analyze_audio_file(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Main pipeline entry point for single clip analysis: Runs Multimodal Gemini 3.5 Lite audio analysis.
    """
    filename = os.path.basename(audio_path)
    logger.info(f"=== Starting Multimodal Gemini analysis pipeline for single file '{filename}' ===")

    result, usage_stats = analyze_audio_with_gemini(audio_path)
    
    logger.info(f"=== Successfully completed single analysis for '{filename}' ===")
    return result, usage_stats


def process_batch(folder_path: str) -> Dict[str, Any]:
    """
    Processes an entire folder containing audio clips using the native Gemini Async Batch Job API.
    Submits batch request with 50% discount Batch Tier pricing ($0.15/1M input, $1.25/1M output).
    Includes manifest validation (labels.csv), latency tracking, and cost ceiling verification.
    """
    batch_start_time = time.time()
    logger.info(f"--- Starting Native Gemini Batch Processing for Directory: '{folder_path}' ---")
    
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

    audio_extensions = {".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac"}
    files = os.listdir(folder_path)

    audio_files = [f for f in sorted(files) if os.path.splitext(f)[1].lower() in audio_extensions]
    logger.info(f"Found {len(audio_files)} audio file(s) to process in native Gemini Batch Job mode: {audio_files}")

    full_paths = [os.path.join(folder_path, f) for f in audio_files]

    total_audio_duration_seconds = 0.0
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_batch_cost_usd = 0.0
    results = []

    matched_manifest_count = 0
    unmatched_audio_files = []
    
    for f in audio_files:
        if f in manifest_filenames:
            matched_manifest_count += 1
        else:
            unmatched_audio_files.append(f)

    try:
        # Submit native batch job
        batch_out = process_audio_batch_job(full_paths)
        for fname, pred, usage in batch_out:
            total_audio_duration_seconds += usage.get("audio_duration_seconds", 0.0)
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_candidate_tokens += usage.get("candidate_tokens", 0)
            total_batch_cost_usd += usage.get("cost_usd", 0.0)

            item = {
                "filename": fname,
                "status": "success" if pred is not None else "failed",
                "prediction": pred.model_dump() if pred is not None else None,
                "usage": usage,
                "ground_truth": ground_truth_map.get(fname)
            }
            results.append(item)
    except Exception as ex:
        logger.error(f"Native Gemini Batch Job failed: {ex}. Falling back to sequential processing...")
        # Fallback to single processing if batch API is unavailable
        for idx, fname in enumerate(audio_files, 1):
            file_path = os.path.join(folder_path, fname)
            try:
                pred, usage = analyze_audio_file(file_path)
                total_audio_duration_seconds += usage.get("audio_duration_seconds", 0.0)
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_candidate_tokens += usage.get("candidate_tokens", 0)
                total_batch_cost_usd += usage.get("cost_usd", 0.0)

                item = {
                    "filename": fname,
                    "status": "success",
                    "prediction": pred.model_dump(),
                    "usage": usage,
                    "ground_truth": ground_truth_map.get(fname)
                }
            except Exception as item_ex:
                item = {
                    "filename": fname,
                    "status": "failed",
                    "error": str(item_ex),
                    "prediction": None,
                    "usage": None,
                    "ground_truth": ground_truth_map.get(fname)
                }
            results.append(item)

    total_batch_latency_sec = round(time.time() - batch_start_time, 2)
    logger.info(f"--- Native Gemini Batch Job Complete: {len(results)} items processed in {total_batch_latency_sec}s ---")
    
    cost_per_audio_min = round((total_batch_cost_usd / (total_audio_duration_seconds / 60)) if total_audio_duration_seconds > 0 else 0.0, 6)
    is_compliant = cost_per_audio_min <= COST_CEILING_PER_MIN_USD

    return {
        "total_files": len(results),
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
        "is_batch_tier": True,
        "results": results
    }

