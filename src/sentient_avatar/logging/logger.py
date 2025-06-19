import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    """Structured logger with JSON formatting and log rotation."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """Initialize structured logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create formatters
        json_formatter = JsonFormatter()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log", maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(json_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log message with extra data.

        Args:
            level: Logging level
            message: Log message
            extra: Extra data to include in log
        """
        if extra is None:
            extra = {}

        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message.

        Args:
            message: Log message
            extra: Extra data to include in log
        """
        self._log(logging.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message.

        Args:
            message: Log message
            extra: Extra data to include in log
        """
        self._log(logging.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message.

        Args:
            message: Log message
            extra: Extra data to include in log
        """
        self._log(logging.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message.

        Args:
            message: Log message
            extra: Extra data to include in log
        """
        self._log(logging.ERROR, message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message.

        Args:
            message: Log message
            extra: Extra data to include in log
        """
        self._log(logging.CRITICAL, message, extra)

    def exception(
        self,
        message: str,
        exc_info: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log exception message.

        Args:
            message: Log message
            exc_info: Include exception info
            extra: Extra data to include in log
        """
        if extra is None:
            extra = {}

        self.logger.exception(message, exc_info=exc_info, extra=extra)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted log record
        """
        # Create base log object
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "extra"):
            log_obj.update(record.extra)

        return json.dumps(log_obj)


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> StructuredLogger:
    """Get structured logger instance.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Structured logger instance
    """
    return StructuredLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
