import logging
from datetime import datetime


class InfoOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.INFO


def setup_logger() -> logging.Logger:
    log_filename = datetime.now().strftime(f"logs/app_%Y-%m-%d_%H-%M-%S.log")

    # Create logger
    logger = logging.getLogger("my_app")
    logger.setLevel(logging.DEBUG)  # Master level must be lowest (DEBUG)

    # --- File Handler (INFO and above) ---
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(InfoOnlyFilter())  # Discard anything that isn't INFO

    # --- Console Handler (DEBUG and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # --- Different Formatters ---
    file_fomratter = logging.Formatter("%(message)s")
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_fomratter)
    console_handler.setFormatter(console_formatter)

    # --- Attach Handlers ---
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger