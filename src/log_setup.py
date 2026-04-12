"""Centralized logging configuration for the thesis pipeline.

Sets up dual-output logging:
  - File handler  → results/logs/<run_name>_<timestamp>.log  (persists in Docker volume)
  - Stream handler → stderr  (visible via ``docker logs`` and compatible with tqdm)

Usage in any entry-point script::

    from src.log_setup import setup_logging

    def main():
        setup_logging("evaluate")   # creates results/logs/evaluate_2026-04-12_18-30-00.log
        ...

The file handler writes to a mounted volume so logs survive container restarts.
Each invocation creates a new log file with a timestamp, making it easy to
correlate logs with specific pipeline runs.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "results" / "logs"

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    run_name: str = "pipeline",
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    stderr_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """Configure root logger with file + stderr handlers.

    Args:
        run_name: Short identifier for this run (e.g. "evaluate", "generate",
            "factuality").  Used as the log-file prefix.
        log_dir: Directory for log files.  Defaults to ``results/logs/``.
            Created automatically if it does not exist.
        level: Root logger level.  Defaults to INFO.
        stderr_level: Minimum level for stderr output.  Defaults to INFO.
        file_level: Minimum level for the log file.  Defaults to DEBUG
            (captures everything, useful for post-mortem analysis).

    Returns:
        Path to the created log file.
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{run_name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. in tests)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── File handler (full detail, survives container crash) ──────────────
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Stderr handler (human-readable, visible in docker logs) ──────────
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(stderr_level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Reduce noise from chatty third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    root.info("Logging → %s", log_file)
    return log_file
