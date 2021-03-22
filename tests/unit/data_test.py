# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import random

import pandas as pd
import pytest

from predictsignauxfaibles.data import SFDataset
import predictsignauxfaibles.config as config


@pytest.fixture
def fake_testing_dataset():
    fake_dta = pd.DataFrame(
        {
            "siren": [round(random.random() * 10 ** 9) for _ in range(20)],
            "output": [random.random() > 0.5 for _ in range(20)],
            "time_til_outcome": [10] * 13 + [0] * 4 + [None] * 2 + [-2] * 1,
            "my_feature": [None] * 9 + [2] * 11,
        }
    )
    dataset = SFDataset()
    dataset.data = fake_dta
    return dataset


def test_full_drop_na(fake_testing_dataset):
    """
    must drop 11 (9 in my_feature, 2 other in time_til_outcome)
    """
    fake_testing_dataset._remove_na(ignore=[])
    assert len(fake_testing_dataset) == 9


def test_part_drop_na(fake_testing_dataset):
    """
    must drop 9 (9 in my_feature, nothing from time_til_outcome, as the field is in ignore)
    """
    fake_testing_dataset._remove_na(ignore=["time_til_outcome"])
    assert len(fake_testing_dataset) == 11


def test_fill_defaults_and_drop_na(fake_testing_dataset):
    """
    must replace 9 (None->0 for my_feature) and drop 2 (from time_til_outcome)
    """
    fake_testing_dataset._replace_missing_data(defaults_map={"my_feature": 0})
    fake_testing_dataset._remove_na(ignore=[])
    assert len(fake_testing_dataset) == 18


def test_remove_strong_signals(fake_testing_dataset):
    """
    must drop 5 (4 0s and 1 (-2)s)
    """
    fake_testing_dataset._remove_strong_signals()
    assert len(fake_testing_dataset) == 15


def test_prepare_data(fake_testing_dataset):
    """
    must drop 11 (NAs) and then 5 more (strong signals)
    must reset index
    """
    fake_testing_dataset.prepare_data(cols_ignore_na=[])
    assert len(fake_testing_dataset) == 9
    fake_testing_dataset.prepare_data(remove_strong_signals=True)
    assert len(fake_testing_dataset) == 4
    assert (fake_testing_dataset.data.index == list(range(4))).all()
