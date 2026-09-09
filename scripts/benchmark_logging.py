"""Shared logging utilities for long-running benchmark scripts."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable


def configure_run_logger(
    output_dir: Path,
    run_name: str,
    *,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Log to both the console and a named file under ``output_dir``.

    Python warnings are routed through the same handlers, so numerical
    fallback warnings remain available after an unattended run finishes.
    Repeated calls in one process do not install duplicate handlers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(log_filename).name != log_filename:
        raise ValueError("log_filename must be a plain filename.")
    log_path = (output_dir / log_filename).resolve()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_console = any(
        getattr(handler, "_cellot_benchmark_console", False)
        for handler in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        console._cellot_benchmark_console = True  # type: ignore[attr-defined]
        root.addHandler(console)

    has_file = any(
        Path(getattr(handler, "baseFilename", "")).resolve() == log_path
        for handler in root.handlers
        if getattr(handler, "baseFilename", None)
    )
    if not has_file:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")
    logger = logging.getLogger(run_name)
    logger.info("run-start | output_dir=%s", output_dir.resolve())
    return logger


def log_terminal_warning(
    logger: logging.Logger,
    *,
    method: str,
    context: str,
    status: str,
    inner_converged: bool = True,
    outer_converged: bool = True,
    cycle_detected: bool = False,
) -> None:
    """Emit one structured warning when a finite terminal result is uncertified."""
    if inner_converged and outer_converged and not cycle_detected:
        return
    logger.warning(
        "terminal-result-retained | method=%s | context=%s | status=%s | "
        "inner_converged=%s | outer_converged=%s | cycle_detected=%s",
        method,
        context,
        status,
        inner_converged,
        outer_converged,
        cycle_detected,
    )


def run_with_exception_logging(entrypoint: Callable[[], None], run_name: str) -> None:
    """Run a script entry point and log an uncaught exception with traceback."""
    try:
        entrypoint()
    except SystemExit as error:
        # argparse uses SystemExit(0) for --help.  It is normal control flow,
        # not a failed benchmark and must not emit an error-level traceback.
        if error.code in (None, 0):
            raise
        logging.getLogger(run_name).exception("run-failed-with-system-exit")
        raise
    except BaseException:
        logging.getLogger(run_name).exception("run-failed-with-uncaught-exception")
        raise
