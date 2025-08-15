import logging
import sys

formatter = logging.Formatter('%(asctime)s - %(message)s')


def add_console_handler(logger: logging.Logger, level):
    global formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return handler


def print_separation_line(logger: bool = False):
    separation_line: str = "=" * 80
    if logger:
        logging.info(separation_line)
    else:
        print(separation_line)
