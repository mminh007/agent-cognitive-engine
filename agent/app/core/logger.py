# app/core/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from app.core.settings import settings

def setup_app_logger(name: str) -> logging.Logger:
    """
    Initializes a highly resilient structural logger utilizing a dual-sink output framework:
    1. StreamHandler (Console Output for instantaneous Docker Telemetry tracking)
    2. RotatingFileHandler (Disk logging bounded by dynamic byte-size rotation triggers)
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if the logger is fetched multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    
    # Resolve physical directories directly mapped from environment settings
    log_directory = settings.logs.dir
    os.makedirs(log_directory, exist_ok=True)
    log_filepath = os.path.join(log_directory, "agent_platform.log")
    
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] [Thread:%(thread)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Stream Handler setup for active console emission
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Rotating File Handler deployment
    file_handler = RotatingFileHandler(
        filename=log_filepath,
        maxBytes=settings.logs.max_bytes,
        backupCount=settings.logs.backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    return logger