@echo off
cd /d "%~dp0"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PYTHONDONTWRITEBYTECODE=1"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m drawinglab.server
pause
