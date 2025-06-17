import logging
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from src.config import settings

def setup_logger():
    # Create logger
    logger = logging.getLogger("sentient_avatar")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10485760,  # 10MB
        backupCount=5
    )

    # Create formatters
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(json_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# Example usage:
# from src.logger import logger
# logger.info("This is an info message")
# logger.error("This is an error message", extra={"error_code": 500}) 