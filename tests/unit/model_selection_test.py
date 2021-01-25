# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import random

import pandas as pd
import pytest

from lib.model_selection import make_sf_test_validate_splits

random.seed(19)


@pytest.fixture
def fake_dataframes():
    fake_siren = [round(random.random() * 10 ** 9) for _ in range(100)]

    train = pd.DataFrame(
        {
            "siren": random.choices(fake_siren, k=200),
            "output": [random.random() > 0.9 for _ in range(200)],
        }
    )
    validate = pd.DataFrame(
        {
            "siren": random.choices(fake_siren, k=100),
            "output": [random.random() > 0.9 for _ in range(100)],
        }
    )

    return train, validate


def test_no_siren_overlap(fake_dataframes):
    splits = make_sf_test_validate_splits(
        fake_dataframes[0], fake_dataframes[1], num_folds=3
    )
    for split in splits.values():
        assert set(fake_dataframes[0].iloc[split["train_on"]].siren).isdisjoint(
            fake_dataframes[1].iloc[split["validate_on"]].siren
        )


@pytest.mark.parametrize("n_folds", [1, 2, 15, 100])
def test_splits_for_different_values(fake_dataframes, n_folds):
    splits = make_sf_test_validate_splits(
        fake_dataframes[0], fake_dataframes[1], num_folds=n_folds
    )
    assert len(splits) == n_folds


@pytest.mark.parametrize("n_folds", [0, -15, 101, 3.5])
def test_value_errors(fake_dataframes, n_folds):
    with pytest.raises(ValueError):
        make_sf_test_validate_splits(
            fake_dataframes[0], fake_dataframes[1], num_folds=n_folds
        )
