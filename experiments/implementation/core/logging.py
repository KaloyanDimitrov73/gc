import logging
from tqdm import tqdm


class TqdmHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


class ExperimentsLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            h = TqdmHandler()
            h.setFormatter(logging.Formatter(
                "\033[32m%(asctime)s\033[0m [%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    def info(self, msg, *a, **kw):
        self.logger.info(msg, *a, **kw)

    def debug(self, msg, *a, **kw):
        self.logger.debug(msg, *a, **kw)

    def warning(self, msg, *a, **kw):
        self.logger.warning(msg, *a, **kw)

    def error(self, msg, *a, **kw):
        self.logger.error(msg, *a, **kw)

    def set_log_file_location(self, path: str):
        ...


def get_logger(name: str) -> ExperimentsLogger:
    return ExperimentsLogger(name)
