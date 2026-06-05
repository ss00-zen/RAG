import logging
import os

LOG_PATH = os.path.join(os.getcwd(), "rag_app.log")

logger = logging.getLogger("rag")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False

logger.debug(f"Logger initialized. Logs writing to {LOG_PATH}")
