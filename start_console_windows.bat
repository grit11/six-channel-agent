@echo off
rem Run the real FastAPI console in the existing langgraph environment.
setlocal
cd /d "%~dp0"
call conda run --no-capture-output -n langgraph python scripts/start_console.py
if errorlevel 1 pause
endlocal
