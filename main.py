# main.py
import os
import json
import shutil
import zipfile
import tempfile
import asyncio
import sys
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load .env file before anything else
load_dotenv()

from pipeline.runner import analyze_audio_file, process_batch
from pipeline.logger import get_logger

# Silence unhandled asyncio background task warnings from SDK cleanup
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def custom_exception_handler(loop, context):
    exception = context.get("exception")
    if isinstance(exception, AttributeError) and "_async_httpx_client" in str(exception):
        return  # Suppress SDK destructor cleanup warning
    loop.default_exception_handler(context)

app = FastAPI(title="EchoPulse AI - Multimodal Gemini Voice Analytics System")

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(custom_exception_handler)

logger = get_logger("EchoPulseAPI")

# Active batch memory cache
CURRENT_BATCH_RESULTS = None


def _load_users() -> Dict[str, str]:
    """
    Parses ECHOPULSE_USERS from the environment into a {username: password} dict.
    Format: "user1:pass1,user2:pass2"
    Falls back to an empty dict (all logins rejected) if the var is missing.
    """
    raw = os.getenv("ECHOPULSE_USERS", "").strip()
    if not raw:
        logger.warning(
            "ECHOPULSE_USERS env var is not set — all login attempts will be rejected. "
            "Add ECHOPULSE_USERS=user:pass,user2:pass2 to your .env file."
        )
        return {}
    users: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            username, _, password = entry.partition(":")
            users[username.strip()] = password.strip()
        else:
            logger.warning(f"Skipping malformed ECHOPULSE_USERS entry (missing ':'): '{entry}'")
    logger.info(f"Loaded {len(users)} user(s) from ECHOPULSE_USERS: {list(users.keys())}")
    return users


@app.get("/", response_class=HTMLResponse)
def index_page():
    logger.info("Serving EchoPulse Light Theme Dashboard HTML index page")
    if os.path.exists("web/index.html"):
        with open("web/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>EchoPulse Dashboard File Not Found</h1>"


@app.get("/api/health")
def health_check():
    logger.info("Health check endpoint pinged")
    return {"status": "ok", "system": "EchoPulse Gemini 3.5 Lite Multimodal Engine", "version": "2.0.0"}


@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    logger.info(f"Login attempt for user: '{username}'")
    users = _load_users()
    expected_password = users.get(username)
    if expected_password is not None and expected_password == password:
        logger.info(f"Successful login for user: '{username}'")
        return {"status": "success", "token": f"echopulse-token-{username}"}
    logger.warning(f"Failed login attempt for user: '{username}'")
    raise HTTPException(status_code=401, detail="Invalid credentials. Check your .env ECHOPULSE_USERS configuration.")


@app.post("/api/config_key")
def config_key(api_key: str = Form(...)):
    os.environ["GEMINI_API_KEY"] = api_key.strip()
    logger.info("Updated GEMINI_API_KEY environment variable.")
    return {"status": "success", "message": "Gemini API Key updated successfully"}


@app.post("/api/upload_batch")
async def upload_batch(file: UploadFile = File(...)):
    global CURRENT_BATCH_RESULTS
    logger.info(f"Batch upload request received: filename='{file.filename}'")
    
    if not file.filename.endswith((".zip", ".tar", ".gz")):
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in {".ogg", ".wav", ".mp3", ".flac", ".m4a", ".aac"}:
            logger.info(f"Processing single audio clip upload with Gemini: '{file.filename}'")
            try:
                pred, usage = analyze_audio_file(file_path)
                CURRENT_BATCH_RESULTS = {
                    "total_files": 1,
                    "total_audio_duration_seconds": usage.get("audio_duration_seconds", 0.0),
                    "total_audio_duration_formatted": usage.get("audio_duration_formatted", "0s"),
                    "total_prompt_tokens": usage.get("prompt_tokens", 0),
                    "total_candidate_tokens": usage.get("candidate_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "total_batch_cost_usd": usage.get("cost_usd", 0.0),
                    "results": [{
                        "filename": file.filename,
                        "status": "success",
                        "prediction": pred.model_dump(),
                        "usage": usage,
                        "ground_truth": None
                    }]
                }
                return CURRENT_BATCH_RESULTS
            except Exception as e:
                logger.error(f"Error processing '{file.filename}' with Gemini: {e}")
                raise HTTPException(status_code=500, detail=f"Gemini Analysis Failed: {str(e)}")
        else:
            logger.error(f"Unsupported upload format: '{file.filename}'")
            raise HTTPException(status_code=400, detail="Please upload an audio file (.mp3, .ogg, .wav, .m4a) or a .zip archive")

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, file.filename)
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    logger.info(f"Extracting ZIP file archive to temporary folder: '{temp_dir}'")
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    target_dir = extract_dir
    subentries = os.listdir(extract_dir)
    if len(subentries) == 1 and os.path.isdir(os.path.join(extract_dir, subentries[0])):
        target_dir = os.path.join(extract_dir, subentries[0])

    logger.info(f"ZIP extracted successfully. Triggering Gemini batch runner on '{target_dir}'")
    CURRENT_BATCH_RESULTS = process_batch(target_dir)
    return CURRENT_BATCH_RESULTS


@app.get("/api/results")
def get_results():
    global CURRENT_BATCH_RESULTS
    logger.info("Fetching batch results...")
    if CURRENT_BATCH_RESULTS is None:
        if os.path.exists("test_docs"):
            logger.info("Loading initial batch from local 'test_docs' directory...")
            CURRENT_BATCH_RESULTS = process_batch("test_docs")
        else:
            return {
                "total_files": 0,
                "total_audio_duration_seconds": 0.0,
                "total_audio_duration_formatted": "0s",
                "total_prompt_tokens": 0,
                "total_candidate_tokens": 0,
                "total_tokens": 0,
                "total_batch_cost_usd": 0.0,
                "results": []
            }
    return CURRENT_BATCH_RESULTS


@app.get("/api/export/csv")
def export_csv():
    global CURRENT_BATCH_RESULTS
    logger.info("Exporting batch results as CSV format")
    if not CURRENT_BATCH_RESULTS:
        raise HTTPException(status_code=400, detail="No batch analysis results available to export")

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["name", "result_json"])

    for item in CURRENT_BATCH_RESULTS.get("results", []):
        name = item["filename"]
        pred_json = json.dumps(item.get("prediction", {}))
        writer.writerow([name, pred_json])

    response = HTMLResponse(content=output.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=echopulse_gemini_predictions.csv"
    return response


@app.get("/api/export/json")
def export_json():
    global CURRENT_BATCH_RESULTS
    logger.info("Exporting batch results as JSON format")
    if not CURRENT_BATCH_RESULTS:
        raise HTTPException(status_code=400, detail="No batch analysis results available to export")

    return JSONResponse(
        content=CURRENT_BATCH_RESULTS,
        headers={"Content-Disposition": "attachment; filename=echopulse_gemini_predictions.json"}
    )