@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" -Desktop
set "DEXHAND_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%DEXHAND_EXIT_CODE%"=="0" echo Shortcut creation failed. See the message above.
pause
exit /b %DEXHAND_EXIT_CODE%
