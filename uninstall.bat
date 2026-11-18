@echo off
rem Remove U-REC from this Windows computer.
rem   uninstall.bat           remove everything U-REC created (keeps the source code)
rem   uninstall.bat --all     also delete this U-REC folder itself
rem   uninstall.bat --yes     don't ask for confirmation (can be combined with --all)
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "ROOT=%CD%"
set "FAILED=0"

rem Pause before closing when started by double-click, so the result stays readable.
set "PAUSE_AT_END=0"
echo %cmdcmdline% | find /i "%~nx0" >nul && set "PAUSE_AT_END=1"

rem Safety: only ever run from a real U-REC folder.
if not exist "cli.py" goto :notrepo
if not exist "urec\config.py" goto :notrepo

set "ALL=0"
set "YES=0"
:args
if "%~1"=="" goto :argsdone
if /I "%~1"=="--all" set "ALL=1" & shift & goto :args
if /I "%~1"=="--yes" set "YES=1" & shift & goto :args
if /I "%~1"=="-y"    set "YES=1" & shift & goto :args
echo Unknown option: %~1
echo Usage: uninstall.bat [--all] [--yes]
set "FAILED=2"
goto :end
:argsdone

rem Where models live (same rules as urec\models.py).
set "CUSTOM_MODELS=0"
set "MODELDIR=%USERPROFILE%\.urec\models"
if defined UREC_MODEL_DIR set "CUSTOM_MODELS=1"
if defined UREC_MODEL_DIR set "MODELDIR=%UREC_MODEL_DIR%"
set "IFDIR=%USERPROFILE%\.insightface"

rem The database may have been moved outside the project in config.toml; ask U-REC where it is.
set "DB="
if exist ".venv\Scripts\python.exe" for /f "usebackq delims=" %%P in (`.venv\Scripts\python.exe -c "from urec import config; print(config.database_path(config.load()))" 2^>nul`) do set "DB=%%P"

echo.
echo U-REC uninstall. This permanently deletes:
echo   Python environment    %ROOT%\.venv
echo   Face database         %ROOT%\data
if defined DB echo   Database file         %DB%
echo   Settings              %ROOT%\config.toml
echo   Python caches         __pycache__ folders in %ROOT%
if "%CUSTOM_MODELS%"=="0" echo   Downloaded models     %USERPROFILE%\.urec
if "%CUSTOM_MODELS%"=="1" echo   Downloaded models     U-REC .onnx files in %MODELDIR%
echo   InsightFace models    %IFDIR%   (shared with any other app that uses InsightFace)
if "%ALL%"=="1" echo   The U-REC folder      %ROOT%   (source code included)
echo.

if "%YES%"=="1" goto :go
set "ANS="
set /p "ANS=Type YES to continue: "
if /I "%ANS%"=="YES" goto :go
echo Cancelled. Nothing was removed.
goto :end

:go
echo.
call :deldir "%ROOT%\.venv"
if defined DB call :delfile "%DB%"
if defined DB call :delfile "%DB%-wal"
if defined DB call :delfile "%DB%-shm"
if defined DB call :delfile "%DB%-journal"
call :deldir "%ROOT%\data"
call :deldir "%ROOT%\dataset"
call :delfile "%ROOT%\config.toml"
call :deldir "%ROOT%\.pytest_cache"
for /d /r "%ROOT%" %%D in (__pycache__) do if exist "%%D\" rd /s /q "%%D" 2>nul
if "%CUSTOM_MODELS%"=="0" call :deldir "%USERPROFILE%\.urec"
if "%CUSTOM_MODELS%"=="1" call :delmodels
call :deldir "%IFDIR%"

echo.
if "%FAILED%"=="1" goto :partial
echo U-REC's environment, face data, settings and models are removed.
if "%ALL%"=="1" goto :selfdelete
echo The source code is still in %ROOT%
echo Delete that folder yourself, or run: uninstall.bat --all
goto :end

:partial
echo Some items could not be removed. Close U-REC and any terminal or editor
echo using this folder, then run uninstall.bat again.
goto :end

:selfdelete
echo Deleting %ROOT% ...
if "%PAUSE_AT_END%"=="1" pause
cd /d "%TEMP%"
rem "(goto)" ends this script first, so the folder holding it can be deleted.
(goto) 2>nul & cd /d "%TEMP%" & rd /s /q "%ROOT%" & if not exist "%ROOT%\" (echo U-REC is completely removed.) else (echo Could not delete %ROOT% - delete it manually.)

:notrepo
echo uninstall.bat must stay inside the U-REC folder (next to cli.py). Nothing was removed.
set "FAILED=2"
goto :end

:end
if "%PAUSE_AT_END%"=="1" pause
exit /b %FAILED%

rem ---- helpers -------------------------------------------------------------

:deldir
if not exist "%~1\" exit /b 0
rd /s /q "%~1" 2>nul
if not exist "%~1\" echo   removed   %~1& exit /b 0
echo   FAILED    %~1
set "FAILED=1"
exit /b 0

:delfile
if not exist "%~1" exit /b 0
del /f /q "%~1" 2>nul
if not exist "%~1" echo   removed   %~1& exit /b 0
echo   FAILED    %~1
set "FAILED=1"
exit /b 0

:delmodels
rem A custom UREC_MODEL_DIR may hold other files, so remove only U-REC's own models.
for %%F in (face_detection_yunet_2023mar face_recognition_sface_2021dec yolox_nano yolox_tiny yolox_s yolox_m yolox_l) do (
  call :delfile "%MODELDIR%\%%F.onnx"
  call :delfile "%MODELDIR%\%%F.onnx.part"
)
rd "%MODELDIR%" 2>nul
exit /b 0
