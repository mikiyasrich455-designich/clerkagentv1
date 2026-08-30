@echo off
rem Clerk local launcher - loads .env and starts the Vercel-native API.
cd /d "%~dp0"
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8010"
python -m uvicorn api.index:app --host 127.0.0.1 --port 8010
