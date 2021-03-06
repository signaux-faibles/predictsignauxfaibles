# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
import random

import pandas as pd
import pytest

from predictsignauxfaibles.model_selection import make_sf_train_test_splits

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
    splits = make_sf_train_test_splits(
        fake_dataframes[0], fake_dataframes[1], num_folds=3
    )
    for split in splits.values():
        assert set(fake_dataframes[0].iloc[split["train_on"]].siren).isdisjoint(
            fake_dataframes[1].iloc[split["test_on"]].siren
        )


def test_validationset_no_overlap_acrossfolds(fake_dataframes):
    splits = make_sf_train_test_splits(
        fake_dataframes[0], fake_dataframes[1], num_folds=4
    )
    for fold_id1 in range(4):
        for fold_id2 in range(fold_id1 + 1, 4):
            assert set(splits[fold_id1]["test_on"]).isdisjoint(
                set(splits[fold_id2]["test_on"])
            )


@pytest.mark.parametrize("n_folds", [1, 2, 15, 100])
def test_splits_for_different_values(fake_dataframes, n_folds):
    splits = make_sf_train_test_splits(
        fake_dataframes[0], fake_dataframes[1], num_folds=n_folds
    )
    assert len(splits) == n_folds


@pytest.mark.parametrize("n_folds", [0, -15, 101, 3.5])
def test_value_errors(fake_dataframes, n_folds):
    with pytest.raises(ValueError):
        make_sf_train_test_splits(
            fake_dataframes[0], fake_dataframes[1], num_folds=n_folds
        )
