"""Centralized logging configuration for the thesis pipeline.

Sets up dual-output logging:
  - File handler   → results/logs/<run_name>.log  (append-mode, persists in Docker volume)
  - Stream handler → stderr  (visible via ``docker logs`` and compatible with tqdm)

Each metric / entry-point gets one stable log file (e.g. ``expert.log``,
``generate.log``).  Successive runs append to the same file with a visible
separator, so ``grep "2026-04-12" results/logs/expert.log`` instantly finds
the run you need without drowning the directory in timestamped files.

Usage in any entry-point script::

    from src.log_setup import setup_logging

    def main():
        setup_logging("evaluate")   # appends to results/logs/evaluate.log
        ...
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = ROOT / "results" / "logs"

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_RUN_SEPARATOR = (
    "\n"
    "{'=' * 72}\n"
    "  NEW RUN — {name} — {ts}\n"
    "{'=' * 72}\n"
)


def setup_logging(
    run_name: str = "pipeline",
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    stderr_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """Configure root logger with file (append) + stderr handlers.

    Args:
        run_name: Short identifier for this run (e.g. "evaluate", "generate",
            "factuality").  Used as the log-file name.
        log_dir: Directory for log files.  Defaults to ``results/logs/``.
            Created automatically if it does not exist.
        level: Root logger level.  Defaults to INFO.
        stderr_level: Minimum level for stderr output.  Defaults to INFO.
        file_level: Minimum level for the log file.  Defaults to DEBUG
            (captures everything, useful for post-mortem analysis).

    Returns:
        Path to the log file.
    """
    log_dir = log_dir or DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{run_name}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. in tests / Streamlit)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── File handler (append-mode, survives container crash) ─────────────
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # ── Stderr handler (human-readable, visible in docker logs) ──────────
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(stderr_level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # Reduce noise from chatty third-party libraries
    for noisy in (
        "httpx", "openai", "httpcore",
        "urllib3", "transformers", "sentence_transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Visual separator between runs in the log file
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = f"\n{'=' * 72}\n  NEW RUN — {run_name} — {ts}\n{'=' * 72}"
    fh.stream.write(sep + "\n")
    fh.stream.flush()

    root.info("Logging to %s (append-mode)", log_file)
    return log_file
