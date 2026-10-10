@echo off
setlocal
set "DEXHAND_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%DEXHAND_PYTHON%" set "DEXHAND_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DEXHAND_PYTHON%" (
    echo Python environment not found. See real\README.md.
    pause
    exit /b 1
)
"%DEXHAND_PYTHON%" "%~dp0launch_gui.pyw" %*
set "DEXHAND_EXIT_CODE=%ERRORLEVEL%"
echo.
echo Exit code: %DEXHAND_EXIT_CODE%
pause
exit /b %DEXHAND_EXIT_CODE%
