import logging
from logging.handlers import RotatingFileHandler
from app.core.config import settings
from rich.logging import RichHandler
from rich.console import Console
import os
from colorama import Fore, Style


LOG_DIR = settings.LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

# Create Rich console for logging
console = Console()

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelno, "")
        message = super().format(record)
        message = message.replace(record.levelname, color + record.levelname + Style.RESET_ALL)
        
        return message

def create_logger(name: str, filename: str = "app.log", level=None) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Use settings LOG_LEVEL if level not explicitly provided
    if level is None:
        log_level_str = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, log_level_str, logging.INFO)
    
    logger.setLevel(level)
    
    # Prevent duplicate handlers and prevent propagation to root logger
    if logger.handlers:
        return logger
    
    # Prevent propagation to avoid duplicate logs from root logger
    logger.propagate = False

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Set handler level too

    # Use the custom colored formatter
    formatter = ColoredFormatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s")
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)

    # If you want to log to a file, add a file handler as well
    log_dir = "logs"  # Example log directory
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setLevel(level)  # Set handler level too
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"))
    logger.addHandler(file_handler)

    return logger


info_logger = create_logger("info_logger", level=logging.INFO)
error_logger = create_logger("error_logger", level=logging.ERROR)
warning_logger = create_logger("warning_logger", level=logging.WARNING)