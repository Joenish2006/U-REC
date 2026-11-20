@echo off
rem One-time setup for Windows.
rem   setup.bat                 free OpenCV backend
rem   setup.bat --insightface   also install InsightFace (best accuracy, non-commercial models)
setlocal
cd /d "%~dp0"

set "WANT_INSIGHTFACE=0"
if /I "%~1"=="--insightface" set "WANT_INSIGHTFACE=1"

set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
  if not defined PY py -3.11 -c "import sys" >nul 2>nul && set "PY=py -3.11"
)
if not defined PY set "PY=python"
%PY% -c "import sys; sys.exit(sys.version_info < (3, 11))" || (echo Python 3.12 is required. Install it with: winget install -e --id Python.Python.3.12 & goto :fail)

%PY% -m venv .venv || goto :fail
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r requirements.txt || goto :fail

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA GPU found: installing onnxruntime-gpu
  "%VPY%" -m pip uninstall -y onnxruntime
  "%VPY%" -m pip install onnxruntime-gpu || goto :fail
)

"%VPY%" cli.py init || goto :fail
if "%WANT_INSIGHTFACE%"=="1" (
  echo Installing InsightFace. This needs Visual Studio Build Tools with "Desktop development with C++".
  "%VPY%" -m pip install -r requirements-insightface.txt || goto :fail
  "%VPY%" -m pip uninstall -y opencv-python-headless
  "%VPY%" -m pip install --force-reinstall --no-deps opencv-python || goto :fail
  "%VPY%" cli.py set-backend insightface || goto :fail
)

"%VPY%" cli.py download-models || goto :fail
echo.
echo Setup complete. Start with: run.bat
exit /b 0

:fail
echo.
echo Setup failed. Read the error above.
exit /b 1
