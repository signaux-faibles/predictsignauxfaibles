import random

import numpy as np
import pandas as pd
from predictsignauxfaibles.decorators import is_random


@is_random
def make_sf_train_test_splits(
    train_data: pd.DataFrame, test_data: pd.DataFrame, num_folds: int
) -> dict:
    """
    Implement Signaux Faibles' custom cross-validation strategy

    Args:
        train_data: a DataFrame containing the training data
        test_data: a DataFrame containing the testing data
        num_folds: the number of folds to make for cross-validation

    Return:
        a dictionary with the following structure :
            {
                "{fold_number}": {
                    "train_on": [{indices}],
                    "test_on": [{indices}]
                }
            }
    """

    if "siren" not in train_data.columns or "siren" not in test_data.columns:
        raise ValueError("training data must contain a `siren` column")

    if len(train_data) < num_folds or len(test_data) < num_folds:
        raise ValueError("num_folds must be smaller than the size of the smallest set")

    if not isinstance(num_folds, int) or num_folds <= 0:
        raise ValueError("num_folds must be a strictly positive integer")

    # get list of SIREN in training data
    list_of_siren = train_data["siren"].unique()
    # shuffle it (just in case)
    random.shuffle(list_of_siren)
    # Splits SIREN in num_folds folds
    siren_splits_train = np.array_split(list_of_siren, num_folds)

    # Generate output
    out = {}
    for i, siren in enumerate(siren_splits_train):
        out[i] = {"train_on": train_data[~train_data["siren"].isin(siren)].index}
        out[i]["test_on"] = test_data[test_data["siren"].isin(siren)].index

    return out
