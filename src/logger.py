import logging
import os
import sys
import io
from logging.handlers import RotatingFileHandler
from config import LOG_FILE

# 🆕 Define startup log file path
STARTUP_LOG_FILE = os.path.join("logs", "startup.log")

# ================================
# 🧰 Ensure log directories exist
# ================================
log_dir_main = os.path.dirname(LOG_FILE)
if log_dir_main:
    os.makedirs(log_dir_main, exist_ok=True)

log_dir_startup = os.path.dirname(STARTUP_LOG_FILE)
if log_dir_startup:
    os.makedirs(log_dir_startup, exist_ok=True)

# ========================================
# 🪟 Windows Console UTF-8 Safety Handling
# ========================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ================================
# 🧰 Safe Stream Handler
# ================================
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            safe_msg = msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def _get_handler(file_path: str, max_bytes: int, backup_count: int):
    """Create a rotating file handler with UTF-8 encoding."""
    handler = RotatingFileHandler(
        file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(fmt)
    return handler


def _get_console_handler():
    """Create UTF-8 safe console handler."""
    handler = SafeStreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(fmt)
    return handler


def get_logger(name="bot"):
    """
    📊 Main trade logger.
    Uses rotating file handler for bot runtime logs with UTF-8 safety.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.addHandler(_get_handler(LOG_FILE, max_bytes=5_000_000, backup_count=3))
    logger.addHandler(_get_console_handler())
    return logger


def get_startup_logger():
    """
    📝 Dedicated startup logger to log all initial config and environment info.
    Keeps startup logs separate from trading activity logs.
    UTF-8 safe on Windows console.
    """
    logger = logging.getLogger("startup")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.addHandler(_get_handler(STARTUP_LOG_FILE, max_bytes=2_000_000, backup_count=2))
    logger.addHandler(_get_console_handler())
    return logger


# ==========================================================
# 🧠 NEW: Helper for detailed MTF disagreement logging
# ==========================================================
def log_mtf_disagreement(timeframe: str, mtf_trend: str, signal_direction: str):
    """
    Logs a clear and actionable message when MTF filter disagrees with signal direction.

    Args:
        timeframe (str): e.g. "15m" or "1h"
        mtf_trend (str): e.g. "LONG", "SHORT", or "NONE"
        signal_direction (str): e.g. "LONG" or "SHORT"
    """
    logger = get_logger()
    logger.info(
        f"⚔️ MTF {timeframe} disagrees → Trend: {mtf_trend or 'NONE'}, Signal: {signal_direction}"
    )
