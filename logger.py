import logging

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


def set_logging():
    logging.basicConfig(level=LOG_LEVEL,
                        format='%(asctime)s - [%(levelname)s] - %(name)s - : %(message)s in %(pathname)s:%(lineno)d',
                        datefmt="%Y-%m-%dT%H:%M:%S%z",
                        )
