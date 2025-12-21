@echo off
title Binance Futures Parameter Optimizer Loop
cd /d D:\projects\binance-futures_oci\src

echo [INFO] Activating virtual environment...
call ..\venv\Scripts\activate

:loop
echo ======================================================
echo [START] Running param_optimizer.py at %date% %time%
echo ======================================================

python param_optimizer.py

echo [INFO] Completed optimization run at %date% %time%
echo [INFO] Waiting 1 hour before next run...

timeout /t 86400 /nobreak >nul

goto loop
