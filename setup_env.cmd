@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1" %*
set "DEXHAND_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%DEXHAND_EXIT_CODE%"=="0" echo Environment setup failed. See the message above.
pause
exit /b %DEXHAND_EXIT_CODE%
