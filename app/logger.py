import logging
import sys


def setup_logger(name: str = "doc-rag") -> logging.Logger:
    """
    Creates and configures a centralized logger for the application.
    """
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers if the logger is imported in multiple files
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # Output logs to the console (stdout), which is perfect for Docker
        console_handler = logging.StreamHandler(sys.stdout)

        # The 'Sweet Spot' Formatter: Timestamp - Level - [File:Line] - Message
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Prevent logs from bubbling up to the root logger to avoid duplicate prints
        logger.propagate = False

    return logger


# Create a singleton instance to be imported by other files
log = setup_logger()
