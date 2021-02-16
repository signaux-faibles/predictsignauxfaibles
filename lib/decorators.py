import logging
from random import seed
from typing import Callable

from config import RANDOM_SEED


def is_random(fun: Callable):
    """
    Functions annotated with @is_random will be seeded if the RANDOM_SEED env variable is found
    """

    def wrapper(*args, **kwargs):
        if RANDOM_SEED:
            logging.debug(f"Seeding random function with seed {RANDOM_SEED}")
            seed(RANDOM_SEED)
        else:
            logging.debug("Random function is not seeded.")
        return fun(*args, **kwargs)

    return wrapper
