import logging

def set_logging(verbose: int):
    level = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
    fmt = ["[%(levelname)s] %(message)s", 
              "[%(levelname)s] %(message)s", 
              "[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d] %(message)s"][verbose]

    logging.root.setLevel(level)
    for h in logging.root.handlers:
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    