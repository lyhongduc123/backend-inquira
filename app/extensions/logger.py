import logging
from logging.handlers import RotatingFileHandler
from core.config import settings
import os

LOG_DIR = settings.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

def create_logger(name: str, filename: str = "app.log", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if settings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    elif not logger.handlers:
        handler = RotatingFileHandler(
            filename=os.path.join(LOG_DIR, filename),
            maxBytes=1_000_000,  # 1MB
            backupCount=5
        )
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

info_logger = create_logger("info_logger", level=logging.INFO)
error_logger = create_logger("error_logger", level=logging.ERROR)
warning_logger = create_logger("warning_logger", level=logging.WARNING)