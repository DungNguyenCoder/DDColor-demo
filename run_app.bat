@echo off
cd /d "%~dp0"
rem May moi clone ve: chua co .venv hoac chua co trong so thi cai dat truoc
if not exist ".venv\Scripts\python.exe" goto :setup
if not exist "weights\ddcolor_paper\pytorch_model.bin" goto :setup
goto :run
:setup
echo Chua cai dat, chay setup.bat truoc ...
call "%~dp0setup.bat" nopause || exit /b 1
:run
set "PY=.venv\Scripts\python.exe"
echo Dang khoi dong demo tai http://localhost:8501 ...
rem mo trinh duyet sau khoang 6 giay, khi may chu da san sang
start "" /min cmd /c "ping -n 7 127.0.0.1 >nul & start http://localhost:8501"
"%PY%" -m streamlit run src\app\app.py
pause
