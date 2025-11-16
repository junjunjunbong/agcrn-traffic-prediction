"""
Logging utility for AGCRN project
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the entire project

    Args:
        log_file: Path to log file. If None, only console logging is used
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (typically __name__)
        level: Optional logging level override

    Returns:
        Logger instance

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def log_model_info(model, logger: logging.Logger) -> None:
    """
    Log model architecture and parameter count

    Args:
        model: PyTorch model
        logger: Logger instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("="*60)
    logger.info("Model Architecture:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("="*60)


def log_config(config_dict: dict, logger: logging.Logger) -> None:
    """
    Log configuration dictionary

    Args:
        config_dict: Configuration dictionary
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("Configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
