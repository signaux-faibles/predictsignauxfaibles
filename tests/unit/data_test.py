# pylint: disable=missing-function-docstring
from lib.data import SFDataset

import config


def test_dataset_init_default_batch_id():
    dataset = SFDataset()
    assert dataset.batch_id == config.BATCH_ID


def test_dataset_init_explicit_batch_id():
    dataset = SFDataset(batch_id="2021_happy_new_year")
    assert dataset.batch_id == "2021_happy_new_year"
