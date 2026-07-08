import random
import os
import numpy as np
from loguru import logger
from dotenv import load_dotenv

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configures the Loguru logger for console and optional file logging."""
    import sys
    logger.remove() # Remove default configuration
    
    # Console Logger
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File Logger
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="10 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    logger.info("Logging successfully initialized.")

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # If using torch or tensorflow later, set seed here
    logger.info(f"Random seed set to {seed} for reproducibility.")

def load_env_variables() -> None:
    """Loads variables from .env file if it exists."""
    if os.path.exists(".env"):
        load_dotenv(".env")
        logger.info("Environment variables loaded from .env")
    else:
        logger.warning(".env file not found. Falling back to default/system environment variables.")
