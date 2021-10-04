import logging
import os
from random import seed
from typing import Callable


def is_random(fun: Callable) -> Callable:
    """Decorator for functions that imply randomness.

    Functions annotated with `@is_random` will be seeded if the RANDOM_SEED environment
    variable is found.

    Args:
        fun: The function to be decorated.

    Returns:
        The decorated function.

    """
    random_seed = os.getenv("RANDOM_SEED")
    random_seed = int(random_seed) if random_seed is not None else False

    def wrapper(*args, **kwargs):
        if random_seed:
            logging.debug(f"Seeding random function with seed {random_seed}")
            seed(random_seed)
        else:
            logging.debug("Random function is not seeded.")
        return fun(*args, **kwargs)

    return wrapper
