import logging

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG

logging_set = False


def set_logging():
    global logging_set
    if not logging_set:
        logging.basicConfig(level=LOG_LEVEL,
                            format='%(asctime)s - [%(levelname)s] - %(name)s - : %(message)s in %(pathname)s:%(lineno)d',
                            datefmt="%Y-%m-%dT%H:%M:%S%z",
                            )
        logging_set = True


def getLogger():
    set_logging()
    return logging.getLogger()
