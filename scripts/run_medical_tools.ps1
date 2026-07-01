# Medical tools — cổng 8010 (KHÔNG dùng 8000, tránh trùng chatbot_api)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:MEDICAL_TOOLS_BASE_URL = "http://localhost:8010"
python -m uvicorn services.medical_tools.app:app --host 0.0.0.0 --port 8010 --reload
