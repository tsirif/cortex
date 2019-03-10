'''Module for logging

'''

import logging

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex')
logger.propagate = False

file_format = '%(asctime)s:%(name)s[%(levelname)s]: %(message)s\n'
stream_format = '[%(levelname)s:%(name)s]: %(message)s' + ' ' * 40


def set_stream_logger(verbosity):
    global logger

    rlevel = logging.WARNING
    rlstr = 'WARNING'
    if verbosity == 0:
        level = logging.WARNING
        lstr = 'WARNING'
    elif verbosity == 1:
        level = logging.INFO
        lstr = 'INFO'
    elif verbosity >= 2:
        level = logging.DEBUG
        lstr = 'DEBUG'
        vv = verbosity - 2
        if vv == 1:
            rlevel = logging.INFO
            rlstr = 'INFO'
        elif vv >= 2:
            rlevel = logging.DEBUG
            rlstr = 'DEBUG'
    else:
        level = logging.INFO
        lstr = 'INFO'

    logging.basicConfig(format=stream_format, level=rlevel)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(stream_format))
    logger.addHandler(ch)
    logger.info('Setting cortex logging to: %s', lstr)
    logger.info('Setting root logging to: %s', rlstr)


def set_file_logger(file_path):
    global logger
    fh = logging.FileHandler(file_path)
    fh.terminator = ''
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(file_format))
    logger.addHandler(fh)
