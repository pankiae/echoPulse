# EchoPulse AI - Multimodal Gemini Voice Analytics System

An end-to-end multimodal voice analytics and audio segmentation pipeline powered by **Google Gemini Flash** with native **Structured Output** (Pydantic Schema).

---

## 🌟 Key Features

1. **Multimodal Audio Reasoning (Zero Hardcoding)**:
   - Evaluates raw audio clips directly (`.mp3`, `.ogg`, `.wav`, `.flac`, `.m4a`) using Google Gemini Flash.
   - Requires no local heavy model downloads, hardcoded audio classification rules, or DSP thresholding.

2. **Native Structured Output (Pydantic Schema)**:
   - Uses `google-genai` SDK `response_schema` enforcement to guarantee strict JSON output matching the 9 target schema fields:
     - `emotional_tone` (`neutral`, `satisfied`, `frustrated`, `upset`, `distressed`)
     - `emotional_intensity` (`low`, `medium`, `high`)
     - `background_noise_present` (`true` / `false`)
     - `background_noise_type` (dynamic text e.g., `"TV"`, `"office chatter"`, `"sharp static"`, `"traffic"`, `"siren"`)
     - `background_noise_severity` (`none`, `low`, `medium`, `high`)
     - `audio_quality` (`clear`, `slightly_impaired`, `severely_impaired`)
     - `speaker_overlap_present` (`true` / `false`)
     - `long_silence_present` (`true` / `false`)
     - `confidence` (`0.0` - `1.0`)

3. **Modern Light-Theme Web UI Dashboard**:
   - Clean, enterprise light-theme dashboard (`web/index.html`).
   - Features an inline **Gemini API Key configuration banner**.
   - Supports drag-and-drop single audio files or `.zip` archives containing evaluation clips and `labels.csv`.
   - Includes a **Copy Markdown Table** button, CSV export, and JSON export.

---

## 📁 Repository Structure

```
echoPulse/
├── main.py                    # FastAPI web server and API endpoints
├── pipeline/
│   ├── __init__.py
│   ├── schema.py              # Pydantic schema model for Gemini Structured Output
│   ├── gemini_analyzer.py     # Multimodal Gemini Flash audio reasoning module
│   ├── runner.py              # Batch runner orchestrator
│   └── logger.py              # Timestamped logger module
├── web/
│   └── index.html             # Light-Theme Web UI Dashboard
└── requirements.txt           # Dependency requirements (google-genai, fastapi, pydantic)
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Virtual Environment Setup

Ensure Python 3.9+ is installed.

#### 🪟 On Windows (PowerShell / Command Prompt):

```powershell
# Navigate to project directory
cd echoPulse

# Create virtual environment
python -m venv .venv

# Activate virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

# (Alternative: Command Prompt activation)
# .venv\Scripts\activate.bat

# Install requirements
pip install -r requirements.txt
```

#### 🐧 On Linux / Ubuntu (Bash Terminal):

```bash
# Navigate to project directory
cd /path/to/echoPulse

# Install python venv package if missing (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3-venv python3-pip

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### 2. Set Gemini API Key

You can set your Google Gemini API key as an environment variable or enter it directly on the web dashboard UI banner.

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

# (Optional) Persist in bashrc
echo 'export GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere"' >> ~/.bashrc
source ~/.bashrc
```

---

### 3. Launching the FastAPI Web Server & Dashboard

Run the server on both OS platforms using Uvicorn:

#### 🪟 Windows:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 🐧 Linux / Ubuntu:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
- **Dashboard UI**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 4. Running as a Background Process (Production Service)

#### 🪟 Windows (Background Task):
```powershell
Start-Process -FilePath "uvicorn" -ArgumentList "main:app --host 0.0.0.0 --port 8000" -WindowStyle Hidden
```

#### 🐧 Linux / Ubuntu (Systemd / nohup / pm2):
```bash
# Option A: Run via nohup in background
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > echopulse.log 2>&1 &

# Option B: Create systemd service (/etc/systemd/system/echopulse.service)
sudo bash -c 'cat <<EOF > /etc/systemd/system/echopulse.service
[Unit]
Description=EchoPulse Gemini Voice Analytics API
After=network.target

[Service]
User=$USER
WorkingDirectory=/path/to/echoPulse
Environment="GEMINI_API_KEY=AIzaSyYourActualAPIKeyHere"
ExecStart=/path/to/echoPulse/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF'

# Enable and start systemd service
sudo systemctl daemon-reload
sudo systemctl enable echopulse
sudo systemctl start echopulse
```

---

## 💻 Python CLI Usage

You can also run batch processing directly in Python scripts across Windows and Linux:

```python
from pipeline.runner import analyze_audio_file, process_batch

# Process a single audio file
result = analyze_audio_file("path/to/call.ogg")
print(result.model_dump())

# Process a directory of audio files
batch_results = process_batch("path/to/audio_folder")
print(batch_results)
```

---

## 📊 Structured JSON Output Format

Every analyzed clip returns a validated JSON payload adhering to the required AutoAce schema:

```json
{
  "emotional_tone": "satisfied",
  "emotional_intensity": "medium",
  "background_noise_present": true,
  "background_noise_type": "sharp static",
  "background_noise_severity": "medium",
  "audio_quality": "clear",
  "speaker_overlap_present": true,
  "long_silence_present": false,
  "confidence": 0.95
}
```
