# 1. Define paths
$VENV_PATH = "C:\Users\marti\XAI4LLMsMetaToolkit\xai-llm-router\.venv\Scripts\Activate.ps1"
$APP_DIR   = "C:\Users\marti\XAI4LLMsMetaToolkit\xai-llm-router"

# 2. Env vars
$env:INSEQ_URL = "http://127.0.0.1:8001"

Write-Host "✅ INSEQ_URL=$env:INSEQ_URL"

# 3. Start backend in a separate window (recommended)
Write-Host "🚀 Launching Inseq Backend..."
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "conda run -n xai-inseq python -m uvicorn inseq_service.app:app --host 127.0.0.1 --port 8001 --log-level info"
)

# 4. Run Streamlit
Set-Location $APP_DIR
Write-Host "🎨 Launching Streamlit from $APP_DIR..."

if (Test-Path $VENV_PATH) {
    . $VENV_PATH
    python -m streamlit run Navigator.py
} else {
    Write-Host "❌ Error: Could not find .venv at $VENV_PATH" -ForegroundColor Red
}