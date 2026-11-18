@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo Dang khoi dong demo tai http://localhost:8501 ...
rem mo trinh duyet sau khoang 6 giay, khi may chu da san sang
start "" /min cmd /c "ping -n 7 127.0.0.1 >nul & start http://localhost:8501"
"%PY%" -m streamlit run src\app\app.py
pause
