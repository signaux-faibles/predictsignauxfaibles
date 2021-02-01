import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from lib.data import SFDataset
from lib.models import SFModel


class SFModelEvaluator:
    """
    Evaluate the performances of a Signaux Faible model
    Input:
        model: a SFModel instance (trained or not)
    """

    def __init__(self, model: SFModel):
        self.model = model
        self.train_set = model.dataset
        self.validate_set = None

    def cv_evaluation(self, num_folds: int, validate_set: SFDataset):
        """
        Evaluate a model using custom cross-validation (see the repo's doc for more information)
        Args:
            num_folds = number of folds to use in Cross-validation
            validate_set = a SFDataset containing the validation data

        NB: evaluation will be biased if the validation set is not at least 18 months
         away from the training set.

        Returns:
            A dictionary in the following format:
                {"{fold_0}": score, ..., "{fold_num_folds}": score,}
        """
        cv_splits = make_sf_test_validate_splits(
            self.train_set.data, validate_set.data, num_folds
        )

        scores = {}
        for i, split in cv_splits.items():
            train = self.train_set.data.iloc[split["train_on"]]
            validate = validate_set.data.iloc[split["validate_on"]]
            self.model.X = train[self.model.features]
            self.model.y = train[[self.model.target]]
            predicted_probas = self.model.train().predict_proba(
                validate[self.model.features]
            )
            scores[i] = self.evaluate(validate[[self.model.target]], predicted_probas)

        return scores

    @staticmethod
    def evaluate(y_true, y_score):
        """
        Evaluation metrics used to evaluate our Signaux Faible models.
        """
        return average_precision_score(y_true, y_score)

    def __repr__(self):
        return f"SFModelEvaluator (model : {type(self.model)})"

    def __str__(self):
        return self.__repr__()


def make_sf_test_validate_splits(
    train_data: pd.DataFrame, validate_data: pd.DataFrame, num_folds: int
) -> dict:
    """
    Implement Signaux Faibles' custom cross-validation strategy

    Input:
        train_data: a DataFrame containing the training data
        validate_data: a DataFrame containing the validation data
        num_folds: the number of folds to make for cross-validation

    Return:
        a dictionary with the following structure :
            {
                "{fold_number}": {
                    "train_on": [{indices}],
                    "validate_on": [{indices}]
                }
            }
    """

    if "siren" not in train_data.columns or "siren" not in validate_data.columns:
        raise ValueError("training data must contain a `siren` column")

    if len(train_data) < num_folds or len(validate_data) < num_folds:
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
        out[i] = {"train_on": train_data[train_data["siren"].isin(siren)].index}
        out[i]["validate_on"] = validate_data[~validate_data["siren"].isin(siren)].index
        # TODO: remove "signaux-forts" from validate_on

    return out
