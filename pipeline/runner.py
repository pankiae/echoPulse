# pipeline/runner.py
import os
import json
import pandas as pd
from typing import Dict, Any, Tuple
from pipeline.schema import AudioAnalysisResult
from pipeline.gemini_analyzer import analyze_audio_with_gemini
from pipeline.audio_utils import format_duration_human
from pipeline.logger import get_logger

logger = get_logger("PipelineRunner")


def analyze_audio_file(audio_path: str) -> Tuple[AudioAnalysisResult, Dict[str, Any]]:
    """
    Main pipeline entry point: Runs Multimodal Gemini 3.5 Lite audio analysis
    and outputs validated Pydantic schema alongside audio duration and token/cost usage statistics.
    """
    filename = os.path.basename(audio_path)
    logger.info(f"=== Starting Multimodal Gemini analysis pipeline for '{filename}' ===")

    result, usage_stats = analyze_audio_with_gemini(audio_path)
    
    logger.info(f"=== Successfully completed analysis for '{filename}' ===")
    return result, usage_stats


def process_batch(folder_path: str) -> Dict[str, Any]:
    """
    Processes an entire folder containing audio clips and optional labels.csv manifest.
    Accumulates total audio clip duration, tokens, and dollar cost across the batch.
    """
    logger.info(f"--- Starting Batch Multimodal Analysis for Directory: '{folder_path}' ---")
    
    labels_csv_path = os.path.join(folder_path, "labels.csv")
    ground_truth_map = {}
    
    if os.path.exists(labels_csv_path):
        logger.info(f"Manifest found: Reading ground truth labels from '{labels_csv_path}'")
        try:
            df = pd.read_csv(labels_csv_path)
            for _, row in df.iterrows():
                fname = row['name']
                raw_json = row['result_json']
                if pd.notna(raw_json) and str(raw_json).strip():
                    ground_truth_map[fname] = json.loads(raw_json)
            logger.info(f"Parsed {len(ground_truth_map)} ground truth records from CSV")
        except Exception as e:
            logger.error(f"Error parsing labels.csv: {e}")

    results = []
    audio_extensions = {".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac"}
    files = os.listdir(folder_path)

    audio_files = [f for f in sorted(files) if os.path.splitext(f)[1].lower() in audio_extensions]
    logger.info(f"Found {len(audio_files)} audio file(s) to process: {audio_files}")

    total_audio_duration_seconds = 0.0
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_batch_cost_usd = 0.0

    for idx, fname in enumerate(audio_files, 1):
        logger.info(f"Processing batch item {idx}/{len(audio_files)}: '{fname}'")
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
            logger.info(f"Item {idx}/{len(audio_files)} '{fname}' completed successfully")
        except Exception as ex:
            logger.error(f"Item {idx}/{len(audio_files)} '{fname}' failed with error: {ex}")
            item = {
                "filename": fname,
                "status": "failed",
                "error": str(ex),
                "prediction": None,
                "usage": None,
                "ground_truth": ground_truth_map.get(fname)
            }
        results.append(item)

    logger.info(f"--- Batch Processing Complete: {len(results)} items processed ---")
    return {
        "total_files": len(results),
        "total_audio_duration_seconds": round(total_audio_duration_seconds, 2),
        "total_audio_duration_formatted": format_duration_human(total_audio_duration_seconds),
        "total_prompt_tokens": total_prompt_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_tokens": total_prompt_tokens + total_candidate_tokens,
        "total_batch_cost_usd": round(total_batch_cost_usd, 6),
        "results": results
    }
