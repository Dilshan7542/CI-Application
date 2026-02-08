@echo off
setlocal

REM Go to this .bat file's folder
cd /d "%~dp0"

REM Run streamlit using the venv python
".venv\Scripts\python.exe" -m streamlit run test.py

pause
