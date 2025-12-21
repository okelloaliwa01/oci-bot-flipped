#!/bin/bash
# ====================================================
# Binance/MTF Trading Bot Launcher with Auto-Restart
# ====================================================

BOT_DIR="/home/ubuntu/binance-futures_5m/src"
PYTHON_EXE="/home/ubuntu/anaconda3/envs/py310/bin/python"
LOG_DIR="$BOT_DIR/logs"

mkdir -p "$LOG_DIR"

DATESTAMP=$(date +"%Y%m%d")
LOGFILE="$LOG_DIR/console_$DATESTAMP.log"

echo "[INFO] Starting trading bot loop..."
echo "Logs: $LOGFILE"
echo "Press CTRL+C to stop."

while true
do
    echo "" >> "$LOGFILE"
    echo "[START] Bot started at $(date)" >> "$LOGFILE"

    "$PYTHON_EXE" "$BOT_DIR/bot.py" >> "$LOGFILE" 2>&1

    echo "[WARN] Bot exited or crashed at $(date). Restarting in 10s..." >> "$LOGFILE"
    sleep 10
done
