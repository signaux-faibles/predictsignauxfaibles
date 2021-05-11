import logging
import random

import numpy as np
import pandas as pd
from predictsignauxfaibles.decorators import is_random

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level="INFO")


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


def merge_models(model_list: list):
    # pylint: disable=all
    # As pylint does not seem to handle recursions properly
    """
    Builds a single list of predicted probabilities based on several models,
    listed by decreasing order of priority.
    For a given SIRET, if no prediction is found for the first model in the list,
    a prediction for the next model will be considered, etc
    Arguments:
        model_list: list
            A list of pandas DataFrame containing, at least, the following columns:
            - siren
            - predicted_probability
    """
    merged = model_list.pop()
    for model_id in range(len(model_list)):
        model = model_list.pop()
        merged = pd.merge(
            model,
            merged,
            left_on="siret",
            right_on="siret",
            how="outer",
            suffixes=("_main", "_supp"),
        )
        merged["predicted_probability"] = merged["predicted_probability_main"].fillna(
            merged["predicted_probability_supp"]
        )
        merged = merged[["siret", "predicted_probability"]]
    return merged


def assign_flag(pred: float, t_rouge: float, t_orange: float):
    """
    Levels a risk score between 0 and 1 to a flag,
    based on thresholds t_rouge and t_orange
    """
    if pred > t_rouge:
        return "Alerte seuil F1"
    elif pred > t_orange:
        return "Alerte seuil F2"
    return "Pas d'alerte"


def split_predictions(preds: pd.DataFrame, t_rouge: float, t_orange: float):
    """
    Generates red/orange/green flags based on two thresholds,
    and logs the share of entries per flag level
    """
    assert "predicted_probability" in preds.columns.tolist()
    preds["alert"] = preds["predicted_probability"].apply(
        lambda x: assign_flag(x, t_rouge, t_orange)
    )

    num_rouge = sum(preds["predicted_probability"] > t_rouge)
    num_orange = sum(preds["predicted_probability"] > t_orange)
    num_orange -= num_rouge
    logging.info(f"{num_rouge} rouge ({round(num_rouge/preds.shape[0] * 100, 2)}%)")
    logging.info(f"{num_orange} orange ({round(num_orange/preds.shape[0] * 100, 2)}%)")

    return preds
