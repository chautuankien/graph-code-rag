"""Logging utilities with context-aware fields and idempotent setup.

This module configures structured logging and injects contextual fields
(`corr_id`, `pipeline`, `run_id`) into every log record. It provides:

- A ContextFilter to populate missing fields from contextvars.
- A pipeline_context() manager to scope pipeline/run identifiers.
- setup_logging() to configure console and rotating file handlers safely.
"""

# src/code_graph_rag/utils/logging_setup.py
from __future__ import annotations

import logging
import logging.config
from contextlib import contextmanager
import contextvars
import uuid
from typing import Iterable, Iterator

# ---- Context (thread/async-safe) ----
# WHY: Use contextvars to carry correlation fields across async tasks/threads.
corr_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("corr_id", default="-")
pipeline_var: contextvars.ContextVar[str] = contextvars.ContextVar("pipeline", default="-")
run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="-")

def _short_id() -> str:
    """Return a short hex identifier suitable for log correlation."""
    return uuid.uuid4().hex[:8]


class ContextFilter(logging.Filter):
    """Filter that injects context fields into every ``LogRecord``.

    Attributes:
        fields: Tuple of field names to inject. Supported values are
            "corr_id", "pipeline", and "run_id".
    """

    def __init__(self, fields: Iterable[str] = ("corr_id", "pipeline", "run_id")):
        """Initialize the filter with the set of fields to inject.

        Args:
            fields: Field names to populate on each record if missing.
        """
        super().__init__()
        self.fields = tuple(fields)

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate missing context fields on the log record.

        Returns:
            True. Filtering never drops a record.
        """
        if "corr_id" in self.fields and not hasattr(record, "corr_id"):
            record.corr_id = corr_id_var.get()
        if "pipeline" in self.fields and not hasattr(record, "pipeline"):
            record.pipeline = pipeline_var.get()
        if "run_id" in self.fields and not hasattr(record, "run_id"):
            record.run_id = run_id_var.get()
        return True


def set_corr_id(value: str) -> None:
    """Set the correlation id for the current execution context."""
    corr_id_var.set(value)


def set_pipeline(value: str) -> None:
    """Set the pipeline name for the current execution context."""
    pipeline_var.set(value)


def set_run_id(value: str) -> None:
    """Set the run id for the current execution context."""
    run_id_var.set(value)


@contextmanager
def pipeline_context(pipeline: str, run_id: str | None = None) -> Iterator[dict[str, str]]:
    """Set pipeline and run id for the active scope.

    Args:
        pipeline: Human-readable pipeline identifier.
        run_id: Optional explicit run identifier. If not provided, one is
            generated automatically.

    Yields:
        A dictionary snapshot with keys ``pipeline`` and ``run_id``.

    WHY: Ensures values are always reset even if an exception is raised.
    """
    t_pl = pipeline_var.set(pipeline)
    t_run = run_id_var.set(run_id or _short_id())
    try:
        yield {"pipeline": pipeline_var.get(), "run_id": run_id_var.get()}
    finally:
        run_id_var.reset(t_run)
        pipeline_var.reset(t_pl)

# Idempotent guard to avoid duplicate handlers/double logging.
_CONFIGURED = False

def setup_logging(level: str = "INFO", log_file: str = "app.log", force: bool = False, is_pytest: bool = False) -> None:
    """Configure root logging once, with console and rotating file handlers.

    This function is safe to call from any pipeline or entry point. By default,
    it will not reconfigure logging if handlers already exist.

    Args:
        level: Minimum log level for handlers (e.g., "INFO", "DEBUG").
        log_file: Path to the rotating log file.
        force: If True, reconfigure even if logging seems already configured.
        is_pytest: setup for pytest environment, which uses file logging only.
    """
    global _CONFIGURED
    root = logging.getLogger()

    # Idempotency: if we've configured already and not forcing, do nothing.
    if _CONFIGURED and not force:
        return
    # Respect host frameworks (Streamlit/Uvicorn/Pytest) that may have
    # configured logging. Avoid duplicating handlers unless force=True.
    if root.handlers and not force:
        # Another framework configured logging; don't duplicate handlers.
        _CONFIGURED = True
        return

    # Formatter templates
    # - %(asctime)s: timestamp formatted by datefmt below
    # - %(levelname)s: record level (INFO/DEBUG/...)
    # - %(name)s: logger name (e.g., module path)
    # - %(pipeline)s/%(run_id)s/%(corr_id)s: context fields injected by ContextFilter
    # - %(message)s: the log message
    # - For file logs, include %(filename)s:%(lineno)d for quick source lookup
    fmt_console = "%(asctime)s %(levelname)s %(name)s [p=%(pipeline)s r=%(run_id)s c=%(corr_id)s]: %(message)s"
    fmt_file = (
        "%(asctime)s %(levelname)s %(name)s [p=%(pipeline)s r=%(run_id)s c=%(corr_id)s] "
        "%(filename)s:%(lineno)d: %(message)s"
    )

    # dictConfig schema
    # - version: Required; dictConfig schema version (must be 1).
    # - disable_existing_loggers: Keep library loggers enabled when False.
    # - filters: Declare reusable filters by name (here: "context").
    # - formatters: Named formatters for console and file outputs.
    # - handlers: Output targets (console/file) referencing formatters/filters.
    # - loggers: Named loggers to configure (note: root logger is usually the
    #   top-level key "root"; a logger named "root" is not the same as the
    #   true root. Kept as-is to match current behavior.).
    logging.config.dictConfig({
        "version": 1,  # Schema version.
        "disable_existing_loggers": False,  # Do not silence library loggers.
        "filters": {
            # Register a filter that injects corr_id/pipeline/run_id into records.
            "context": {"()": ContextFilter, "fields": ["corr_id", "pipeline", "run_id"]}
        },
        "formatters": {
            # Console and file formatter. datefmt controls how %(asctime)s is rendered.
            "console": {"format": fmt_console, "datefmt": "%Y-%m-%d %H:%M:%S"},
            "file": {"format": fmt_file, "datefmt": "%Y-%m-%d %H:%M:%S"},
        },
        "handlers": {
            # Stream to stderr by default. Attach context filter and console formatter.
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "console",
                "filters": ["context"],
            },
            # Rotating file handler: ~5MB per file, keep 3 backups, UTF-8 encoding.
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "file",
                "filename": log_file,
                "maxBytes": 5_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "filters": ["context"],
            },
        },
        "loggers": {
            # Configure a logger named "root" with both handlers. Note: in
            # dictConfig, the true root logger is configured via top-level
            # key "root". This entry targets a regular logger named "root".
            "root": {  # root for entire app (named logger)
                # If is_pytest, use only file handler to avoid console noise.
                "handlers": (["file"] if is_pytest else ["console", "file"]),
                "level": level,
            },
        },
    })
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, defaulting to this module's name.

    Args:
        name: Optional logger name. Defaults to the current module name.

    Returns:
        A standard ``logging.Logger`` instance.
    """
    return logging.getLogger(name or __name__)
