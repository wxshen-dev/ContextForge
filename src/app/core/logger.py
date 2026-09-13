"""
Project logging utilities.
Implemented with loguru. Supports console and file output controlled by .env,
and automatically generates logs/app_YYYYMMDD.log.

Features:
1. Configuration-driven output switches and log levels.
2. Automatic file log path under project_root/logs/app_YYYYMMDD.log.
3. Automatic cleanup based on the configured retention policy.
4. UTF-8 encoding for stable multilingual output.
5. Async-safe queued logging for threaded and async scenarios.
6. Ready to use by importing logger from project modules.
7. Accurate caller location by skipping loguru internals and this utility module.
"""
import sys
import inspect
from pathlib import Path
import os
from dotenv import load_dotenv
from loguru import logger


# -------------------------- Step 1: Load .env Configuration --------------------------
load_dotenv()

# -------------------------- Step 2: Read .env Configuration with Defaults --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

# -------------------------- Step 3: Define Log Paths --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE_NAME = "app_{time:YYYYMMDD}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

# -------------------------- Step 4: Define Log Format --------------------------
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
    "<level>{message}</level>"
)

# -------------------------- Step 5: Initialize Logger Configuration --------------------------
def init_logger():
    """
    Initialize the global logger configuration.
    1. Remove loguru's default console sink to avoid duplicate output.
    2. Enable or disable console output based on .env settings.
    3. Enable or disable file output and create the logs directory automatically.
    4. Configure format, level, rotation, and retention.
    :return: Configured loguru logger instance.
    """
    # 1. Remove loguru's default console output.
    logger.remove()

    # 2. Configure console output when enabled.
    if LOG_CONSOLE_ENABLE:
        logger.add(
            sink=sys.stdout,
            level=LOG_CONSOLE_LEVEL,
            format=LOG_FORMAT,
            colorize=True,
            enqueue=True
        )

    # 3. Configure file output when enabled.
    if LOG_FILE_ENABLE:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            sink=LOG_FILE_PATH,
            level=LOG_FILE_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=LOG_FILE_RETENTION,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    return logger

# -------------------------- Step 6: Initialize and Patch the Global Logger --------------------------
base_logger = init_logger()

def fix_log_position(record):
    """Walk the call stack and extract the actual business-code caller location."""
    for frame in inspect.stack():
        # Skip loguru internals and this logger utility module.
        if ("_logger.py" in frame.filename or frame.function == "_log") or "logger.py" in frame.filename:
            continue
        # Update log fields with the business-code caller location.
        record.update(
            name=frame.filename.split("/")[-1].split("\\")[-1],
            function=frame.function,
            line=frame.lineno
        )
        break

# Apply the caller-location patch and export the project-wide logger.
logger = base_logger.patch(fix_log_position)


from functools import wraps
import time
from typing import Mapping

def _trace_id(state) -> str:
    if isinstance(state, Mapping):
        return str(state.get("session_id") or state.get("task_id") or "-")
    return "-"

def node_log(node_name: str):
    def deco(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_id(state)
            start_ts = time.time()
            logger.info(f"[{node_name}] node started, trace_id={trace_id}")
            try:
                result = func(state, *args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{node_name}] node completed, trace_id={trace_id}, elapsed={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{node_name}] node failed, trace_id={trace_id}")
                raise
        return wrapper
    return deco

def step_log(step_name: str):
    """
    Step logging decorator.
    - Logs step start, completion, and exceptions with stack traces.
    - Re-raises exceptions to preserve business semantics.
    """
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ts = time.time()
            logger.info(f"[{step_name}] step started")
            try:
                result = func(*args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{step_name}] step completed, elapsed={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{step_name}] step failed")
                raise
        return wrapper
    return deco
