@echo off
REM ====================================================
REM  Binance/MTF Trading Bot Launcher with Auto-Restart
REM ====================================================

setlocal enabledelayedexpansion

REM --- Adjust these paths ---
set BOT_DIR=D:\projects\binance-futures_5m\src
set PYTHON_EXE=C:\Users\Support\anaconda3\envs\py310\python.exe
set LOG_DIR=%BOT_DIR%\logs
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM --- Daily log file name ---
set DATESTAMP=%DATE:~-4%%DATE:~3,2%%DATE:~0,2%
set LOGFILE=%LOG_DIR%\console_%DATESTAMP%.log

echo [INFO] Starting trading bot loop...
echo Logs: %LOGFILE%
echo Press CTRL+C to stop.

REM --- Infinite loop with restart on crash ---
:loop
    echo. >> "%LOGFILE%"
    echo [START] Bot started at %TIME% on %DATE% >> "%LOGFILE%"
    "%PYTHON_EXE%" "%BOT_DIR%\bot.py" >> "%LOGFILE%" 2>&1

    echo [WARN] Bot exited or crashed at %TIME%. Restarting in 10s... >> "%LOGFILE%"
    timeout /t 10 >nul
goto loop
