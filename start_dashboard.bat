@echo off
REM One-click start for Windows: creates a virtual environment on first run, installs deps, starts the dashboard.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt
start "" /min cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8000"
python scripts\serve.py
pause
