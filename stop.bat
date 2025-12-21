@echo off
echo [INFO] Stopping bot...
taskkill /IM python.exe /F >nul 2>&1
echo [OK] Bot stopped.
pause
