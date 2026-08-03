# EchoPulse AI - Multimodal Gemini Voice Analytics System

An end-to-end multimodal voice analytics and audio segmentation pipeline powered by **Google Gemini 3.5 Flash-Lite** with native **Structured Output** (Pydantic Schema).

---

## 🌟 Key Features

1. **Multimodal Audio Reasoning (Zero Hardcoding)**:
   - Evaluates raw audio clips directly (`.mp3`, `.ogg`, `.wav`, `.flac`, `.m4a`) using Google Gemini Flash-Lite.
   - Requires no heavy local machine learning models or hardcoded audio classification rules.

2. **Native Structured Output (Pydantic Schema)**:
   - Enforces strict JSON output matching the target 9 schema fields:
     - `emotional_tone` (`neutral`, `satisfied`, `frustrated`, `upset`, `distressed`)
     - `emotional_intensity` (`low`, `medium`, `high`)
     - `background_noise_present` (`true` / `false`)
     - `background_noise_type` (dynamic text e.g., `"TV"`, `"office chatter"`, `"sharp static"`, `"line hiss"`)
     - `background_noise_severity` (`none`, `low`, `medium`, `high`)
     - `audio_quality` (`clear`, `slightly_impaired`, `severely_impaired`)
     - `speaker_overlap_present` (`true` / `false`)
     - `long_silence_present` (`true` / `false`)
     - `confidence` (`0.0` - `1.0`)

3. **Cost Optimization**:
   - Runs on Gemini 3.5 Flash-Lite ($0.30/1M input, $2.50/1M output), staying strictly within budget thresholds (<$0.003/min).

4. **Telemetry & Audio Duration Tracking**:
   - Tracks total audio content play duration (`mutagen` header parser) separately from token counts and cost calculation.

5. **Modern Light-Theme Web UI Dashboard**:
   - Clean, enterprise light-theme dashboard (`web/index.html`).
   - Inline **Gemini API Key configuration banner**.
   - Supports drag-and-drop single audio files or `.zip` archives.
   - Includes **Copy Markdown Table**, CSV export, and JSON export buttons.

---

## 📁 Repository Structure

```
echoPulse/
├── main.py                    # FastAPI web server and API endpoints
├── pipeline/
│   ├── __init__.py
│   ├── schema.py              # Pydantic schema model for Gemini Structured Output
│   ├── gemini_analyzer.py     # Multimodal Gemini Flash audio reasoning module
│   ├── audio_utils.py         # Mutagen audio duration calculation module
│   ├── runner.py              # Directory & ZIP runner orchestrator
│   └── logger.py              # Timestamped logger module
├── web/
│   └── index.html             # Light-Theme Web UI Dashboard
└── requirements.txt           # Dependency requirements (google-genai, fastapi, mutagen, pydantic)
```

---

## 🚀 Local Setup Guide

### 1. System Requirements & Virtual Environment Setup

Ensure Python 3.9+ is installed.

#### 🪟 On Windows (PowerShell / Command Prompt):

```powershell
# 1. Navigate to project directory
cd echoPulse

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

# (Alternative: Command Prompt activation)
# .venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements.txt
```

#### 🐧 On Linux / Ubuntu (Bash Terminal):

```bash
# 1. Navigate to project directory
cd /path/to/echoPulse

# 2. Install python venv package if missing (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3-venv python3-pip

# 3. Create virtual environment
python3 -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

---

### 2. Configure Gemini API Key

Set your Google Gemini API key as an environment variable (or enter it directly in the web dashboard banner).

#### 🪟 On Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere"
```

#### 🪟 On Windows (CMD):
```cmd
set GEMINI_API_KEY=AIzaSyYourActualAPIKeyHere
```

#### 🐧 On Linux / Ubuntu (Bash):
```bash
export GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere"
```

---

### 3. Running the Web Server & Dashboard Locally

Start the local FastAPI application server using Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, access the local service in your browser:
- 🌐 **Web Dashboard UI**: [http://localhost:8000/](http://localhost:8000/)
- 📑 **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🔐 **Default Dashboard Credentials**: `admin` / `autoace2026` or `evaluator` / `eval2026`

---

## 💻 Python CLI / Local Script Usage

You can also run analysis directly in local Python scripts:

```python
from pipeline.runner import analyze_audio_file, process_batch

# 1. Process a single audio clip
pred, usage = analyze_audio_file("path/to/sample.mp3")
print("Single Audio Prediction:", pred.model_dump())
print("Usage Stats:", usage)

# 2. Process a directory of audio clips
batch_results = process_batch("path/to/audio_folder")
print(f"Processed {batch_results['total_files']} files. Total Audio: {batch_results['total_audio_duration_formatted']}")
```

---

## 📊 Sample Output Format

Every analyzed clip returns a validated JSON payload matching the required AutoAce schema:

```json
{
  "emotional_tone": "satisfied",
  "emotional_intensity": "medium",
  "background_noise_present": true,
  "background_noise_type": "sharp static",
  "background_noise_severity": "medium",
  "audio_quality": "clear",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.95
}
```
