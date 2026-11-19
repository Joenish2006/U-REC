@echo off
rem Short way on Windows:  run.bat   run.bat identify   run.bat register Joenish
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup.bat first.
  exit /b 1
)
.venv\Scripts\python.exe cli.py %*
