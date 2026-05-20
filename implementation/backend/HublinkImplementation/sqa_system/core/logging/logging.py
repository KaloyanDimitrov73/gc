import logging
from logging import FileHandler
from pprint import pformat
from datetime import datetime
import os
import shutil
import yaml
from tqdm import tqdm

from sqa_system.core.data.file_path_manager import FilePathManager


class CustomFormatter(logging.Formatter):
    """
    Formatter that checks if the message starts with "SEPARATOR"
    if so, it will remove the prefix and return the rest of the message.
    """

    def format(self, record):
        if record.msg.startswith("SEPARATOR"):
            return record.msg.split("SEPARATOR")[1]
        return super().format(record)
    
class TqdmLoggingHandler(logging.Handler):
    """
    We use a custom formatter to print the outputs above the progress bars 
    using the tqdm write method as mentioned here:
    https://medium.lies.io/progress-bar-and-status-logging-in-python-with-tqdm-35ce29b908f5
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name):
    """
    Returns a custom logger.
    """
    return CustomLogger(name)


def _load_logging_config() -> dict:
    file_path_manager = FilePathManager()
    log_config_path = file_path_manager.get_path("logging_config.yaml")

    with open(log_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class LogInitializer:
    """
    A singleton class that initializes the logger for the QASystem.
    This class makes sure, that only one logger is created and that
    the log file is rotated at the start of each session.
    """
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogInitializer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self._is_initialized = True
            self.logger = self.setup_logging()

    def rotate_log_file(self, current_log_path, backup_format, backup_count):
        """
        Rotate the current log file to a backup and create a new empty log file.
        """
        print("Rotating log file")
        if os.path.exists(current_log_path):
            backup_filename = datetime.now().strftime(backup_format)
            backup_path = os.path.join(os.path.dirname(
                current_log_path), backup_filename)

            # Check if file is locked by another process
            try:
                shutil.move(current_log_path, backup_path)
            except PermissionError:
                print("Log file is locked by another process. Skipping rotation.")
                return

            # Delete old backups if we have more than backup_count
            log_dir = os.path.dirname(current_log_path)
            all_logs = sorted([f for f in os.listdir(log_dir) if f.startswith(
                "log_") and f.endswith(".log")], reverse=True)
            for old_log in all_logs[backup_count:]:
                os.remove(os.path.join(log_dir, old_log))

    def setup_logging(self):
        """
        Set up logging for the QASystem.
        """
        config = _load_logging_config()
        logger = logging.getLogger("QASystem")

        if not logger.handlers:
            logger.setLevel(config["log_level"])

            file_path_manager = FilePathManager()
            log_file_path = file_path_manager.get_path(
                config["current_log_filename"])
            file_path_manager.ensure_dir_exists(log_file_path)

            # Rotate the log file at the start of each session
            self.rotate_log_file(
                log_file_path, config["backup_filename_format"], config["backup_count"])

            file_handler = FileHandler(
                filename=log_file_path,
                encoding="utf-8"
            )

            # Add the custom formatter for the log file itself
            formatter = CustomFormatter(config["log_format"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Add the custom TQDM progress bar logging handler for writing INFO level to console
            console_handler = TqdmLoggingHandler()
            console_handler.setLevel(logging.INFO)
            simple_formatter = logging.Formatter(
                 "\033[32m%(asctime)s\033[0m - %(message)s")
            console_handler.setFormatter(simple_formatter)
            logger.addHandler(console_handler)

            # Prevent log messages from being propagated to the root logger
            logger.propagate = False

            logger.info("New session started")
        return logger


class CustomLogger:
    """
    Custom logging wrapper class that extends Pythons built-in logging functionality.

    Args:
        name (str): The name to be appended to "QASystem." prefix in the logger name.
    """

    def __init__(self, name):
        self.logger = logging.getLogger(f"QASystem.{name}")
        self.log_initializer = LogInitializer()

    def debug(self, msg, *args, **kwargs):
        """
        Log a debug message.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.debug(msg, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Logs an information message with the logger.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.info(msg, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Logs a warning message.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.warning(msg, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log an error message with the specified parameters.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.error(msg, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """
        Records exception information with a message, formatting arguments into the message.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.exception(msg, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log a critical level message through the logger.
        """
        msg = self._process_message(msg, *args, **kwargs)
        self.logger.critical(msg, **kwargs)

    def _process_message(self, msg, *args):
        if not args:
            return msg

        formatted_args = []
        for arg in args:
            if isinstance(arg, dict):
                # Pretty-print dictionaries with indentation
                formatted_arg = pformat(arg, indent=4, width=80)
                formatted_args.append(formatted_arg)
            elif isinstance(arg, list):
                if all(isinstance(item, dict) for item in arg):
                    formatted_arg = pformat(arg, indent=4, width=80)
                    formatted_args.append(formatted_arg)
                else:
                    formatted_args.append(arg)
            else:
                formatted_args.append(arg)

        # Attempt to format the message with the formatted arguments
        try:
            if "%" in msg:
                msg = msg % tuple(formatted_args)
            else:
                msg = f"{msg} {" ".join(str(arg) for arg in formatted_args)}"
        except Exception:
            pass

        return msg
    
    def set_log_file_location(self, new_log_path: str):
        """
        Change the log file location.
        
        Args:
            new_log_path (str): The new path for the log file.
        """
        # Remove existing file handlers from the root logger
        parent_logger = logging.getLogger("QASystem")
        for handler in parent_logger.handlers[:]:
            if isinstance(handler, FileHandler):
                parent_logger.removeHandler(handler)
                handler.close()

        # Ensure the directory exists
        new_dir = os.path.dirname(new_log_path)
        os.makedirs(new_dir, exist_ok=True)

        # Create a new FileHandler for the new log file
        file_handler = FileHandler(filename=new_log_path, encoding="utf-8")
        # Load the logging configuration to use the same log format
        config = _load_logging_config()
        formatter = CustomFormatter(config["log_format"])
        file_handler.setFormatter(formatter)
        parent_logger.addHandler(file_handler)
        parent_logger.info("Log file location changed to %s", new_log_path)
