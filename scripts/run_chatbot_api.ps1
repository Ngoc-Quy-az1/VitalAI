# Chatbot API — cổng 8000 (frontend / Vite proxy trỏ vào đây)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m uvicorn services.chatbot_api.app:app --host 0.0.0.0 --port 8000 --reload
