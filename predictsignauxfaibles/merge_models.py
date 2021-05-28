# pylint: disable=all
import logging
from typing import List

import pandas as pd


def merge_models(model_list: List[pd.DataFrame]):
    """
    Builds a single list of predicted probabilities based on several models,
    listed by decreasing order of priority.
    For a given SIRET, if no prediction is found for the first model in the list,
    a prediction for the next model will be considered, etc
    Arguments:
        model_list: list
            A list of pandas DataFrame containing, at least, the following columns: siren, predicted_probability
    """
    try:
        merged = model_list.pop()
    except IndexError:
        logging.error("model_list appears to be empty")
    if model_list == []:
        logging.warning("model_list contains a single model")
        return merged

    for model in model_list:
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
