# pylint: disable=missing-function-docstring
import os
from random import randint

os.environ["RANDOM_SEED"] = "42"  # Manually set env var for testing
from predictsignauxfaibles.decorators import (  # pylint: disable=wrong-import-position
    is_random,
)


def test_is_random_seeded():
    @is_random
    def random_function(mult=1):
        return randint(0, 100) * mult

    ten_function_calls = [random_function() for _ in range(10)]

    assert len(set(ten_function_calls)) == 1


def test_is_random_not_seeded():
    del os.environ["RANDOM_SEED"]

    @is_random
    def random_function(mult=1):
        return randint(0, 100) * mult

    ten_function_calls = [random_function() for _ in range(10)]

    assert len(set(ten_function_calls)) > 1
