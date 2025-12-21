#!/bin/bash
# ======================================================
# Binance Futures Parameter Optimizer Loop
# ======================================================

# --- Configuration ---
PROJECT_DIR="/home/ubuntu/binance-futures_5m"
SRC_DIR="$PROJECT_DIR/src"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_EXE="$VENV_DIR/bin/python"
LOG_DIR="$SRC_DIR/logs"

mkdir -p "$LOG_DIR"

echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

while true
do
    echo "======================================================"
    echo "[START] Running param_optimizer.py at $(date)"
    echo "======================================================"

    "$PYTHON_EXE" "$SRC_DIR/param_optimizer.py" >> "$LOG_DIR/optimizer_$(date +%Y%m%d).log" 2>&1

    echo "[INFO] Completed optimization run at $(date)"
    echo "[INFO] Waiting 24 hours before next run..."
    sleep 86400  # 24 hours
done
