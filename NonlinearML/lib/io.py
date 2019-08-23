import sys
import logging


def setConfig(path):
    """ Set logging config to a file."""
    logging.basicConfig(
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()],
        level=logging.INFO)


def hbar(length):
    """ Draw horizontal bar with given length."""
    return '='*length

def title(title, length=80):
    """ Print title with horizontal bars."""
    logging.info(hbar(length))
    logging.info(" " + title)
    logging.info(hbar(length))

def message(message):
    """ Print message."""
    if type(message) == list:
        for line in message:
            logging.info(" " + line)
    else:
        logging.info(" " + message)

def error(error):
    """ Print error."""
    logging.error("ERROR: " + message)
