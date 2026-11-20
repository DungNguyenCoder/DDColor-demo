@echo off
rem Cai dat lan dau tren may moi: tao .venv, cai PyTorch (GPU neu co NVIDIA, khong thi CPU),
rem cai thu vien trong requirements.txt va tai trong so mo hinh (~2 GB). Chay lai nhieu lan cung duoc.
setlocal
cd /d "%~dp0"

set "SYS="
where py >nul 2>nul && py -3 -c "import sys" >nul 2>nul && set "SYS=py -3"
if not defined SYS where python >nul 2>nul && set "SYS=python"
if not defined SYS goto :nopython
%SYS% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" || goto :oldpython

if exist ".venv\Scripts\python.exe" goto :venv_ok
echo [1/4] Tao moi truong ao .venv ...
%SYS% -m venv .venv || goto :fail
:venv_ok
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip >nul

echo [2/4] Cai PyTorch ...
"%PY%" -c "import torch" >nul 2>nul && goto :torch_ok
where nvidia-smi >nul 2>nul || goto :torch_cpu
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 && goto :torch_ok
echo Khong cai duoc ban GPU, chuyen sang ban CPU.
:torch_cpu
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :fail
:torch_ok

echo [3/4] Cai cac thu vien khac ...
"%PY%" -m pip install -r requirements.txt || goto :fail

echo [4/4] Tai trong so mo hinh (~2 GB, chi tai lan dau) ...
"%PY%" scripts\download_weights.py || goto :fail

echo.
echo Cai dat xong. Nhay dup run_app.bat de mo demo.
if /i not "%~1"=="nopause" pause
exit /b 0

:nopython
echo Khong tim thay Python. Cai Python 3.10-3.13 tu https://www.python.org/downloads/ (tick "Add python.exe to PATH") roi chay lai.
pause
exit /b 1
:oldpython
echo Can Python 3.10 tro len.
pause
exit /b 1
:fail
echo.
echo Cai dat that bai, xem loi o tren.
pause
exit /b 1
