from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from logging.handlers import TimedRotatingFileHandler
import logging
import sys
import os

from ..utils import cfg


class Logger:
    FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = cfg.debugging.log_file
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    @classmethod
    def get_console_handler(cls):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(cls.FORMATTER)
        return console_handler
    
    @classmethod
    def get_file_handler(cls):
        file_handler = TimedRotatingFileHandler(cls.LOG_FILE, when='midnight')
        file_handler.setFormatter(cls.FORMATTER)
        return file_handler
    
    @classmethod
    def get_logger(cls, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            logger.addHandler(cls.get_console_handler())
            logger.addHandler(cls.get_file_handler())

        logger.propagate = False
        return logger
    
if __name__ == "__main__":
    _logger = Logger.get_logger("TRAINING")
    _logger.info("box_loss: 0.1 - conf_loss: 0.1 - cls_loss: 0.1")

