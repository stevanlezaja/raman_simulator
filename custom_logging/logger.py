import logging


# Keep track of all logger names
LOGGER_NAMES = set()

# Save original getLogger function
_original_getLogger = logging.getLogger

def getLogger(name=None):
    logger = _original_getLogger(name)
    LOGGER_NAMES.add(logger.name)
    return logger

# Patch logging.getLogger globally
logging.getLogger = getLogger

# Keep track of logger names
LOGGER_NAMES: set[str] = set()

def get_logger(name: str):
    LOGGER_NAMES.add(name)
    return logging.getLogger(name)

def get_max_name_len():
    return max((len(n) for n in LOGGER_NAMES), default=0)

class DynamicPaddedFormatter(logging.Formatter):
    def format(self, record):
        max_len = max(len(n) for n in LOGGER_NAMES) if LOGGER_NAMES else len(record.name)
        record.name = record.name.ljust(max_len)
        return super().format(record)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        DynamicPaddedFormatter(fmt="[%(name)s] %(levelname)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(handler)
