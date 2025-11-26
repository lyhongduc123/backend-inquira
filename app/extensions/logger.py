import logging
from logging.handlers import RotatingFileHandler
from app.core.config import settings
from rich.logging import RichHandler
from rich.console import Console
import os

LOG_DIR = settings.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# Create Rich console for logging
console = Console()

def create_logger(name: str, filename: str = "app.log", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if settings.LOG_TO_CONSOLE:
        # Use Rich handler for beautiful console logging
        class RedErrorFormatter(logging.Formatter):
            def format(self, record):
                msg = super().format(record)
                if record.levelno == logging.ERROR:
                    msg = msg.replace('ERROR', '[red]ERROR[/red]')
                return msg

        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            show_time=True,
            show_level=True,
            show_path=True,
        )
        console_handler.setFormatter(RedErrorFormatter("%(message)s"))
        logger.addHandler(console_handler)
    elif not logger.handlers:
        # File handler remains the same
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