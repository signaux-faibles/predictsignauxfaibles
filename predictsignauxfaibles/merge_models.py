# pylint: disable=all
import logging
from typing import List

import pandas as pd


def merge_models(model_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Builds a list of predicted probabilities based on several models.

    The available probabilities are picked by decreasing order of priority, that is,
    for a given SIRET, if no prediction is found for the first model in the list, a
    prediction for the next model will be considered, etc.

    Args:
        model_list: A list of pandas DataFrame containing, at least, the following
        columns: 'siren', 'predicted_probability'

    Returns:
        A DataFrame with merged predicted probabilities.

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
