# 1. Define the REAL paths (Adjust these to your machine!)
$VENV_PATH = "C:\Users\marti\XAI4LLMsMetaToolkit\xai-llm-router\.venv\Scripts\Activate.ps1"
$APP_DIR = "C:\Users\marti\Documents\xaimetatool"

# 2. Set Environment Variables
$env:INSEQ_URL = "http://127.0.0.1:8001"

# 3. Start Backend (Conda) in background
echo "🚀 Launching Inseq Backend..."
Start-Job -ScriptBlock {
    conda run -n xai-inseq python -m uvicorn inseq_service.app:app --host 127.0.0.1 --port 8001
} | Out-Null

# 4. Navigate to the App folder and Run Streamlit
cd $APP_DIR
echo "🎨 Launching Streamlit from $APP_DIR..."

if (Test-Path $VENV_PATH) {
    . $VENV_PATH
    python -m streamlit run Navigator.py
} else {
    Write-Host "❌ Error: Could not find .venv at $VENV_PATH" -ForegroundColor Red
}