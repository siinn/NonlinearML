import logging
import sys
import os



def setConfig(path,filename):
    """ Set logging config to a file."""
    if not os.path.exists(path):
        os.makedirs(path)
    # Creat logger
    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    # Create file handler which logs even INFO messages
    fh = logging.FileHandler(path+filename)
    fh.setLevel(logging.INFO)
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s',"%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

def hbar(length):
    """ Draw horizontal bar with given length."""
    return '='*length

def title(title, length=80):
    """ Print title with horizontal bars."""
    logger = logging.getLogger('test')
    logger.info(hbar(length))
    logger.info(" " + title)
    logger.info(hbar(length))

def message(message):
    """ Print message."""
    logger = logging.getLogger('test')
    if type(message) == list:
        for line in message:
            logger.info(" " + line)
    else:
        logger.info(" " + message)

def error(error):
    """ Print error."""
    logger = logging.getLogger('test')
    logger.error(message)
