# pylint: disable=all
from types import ModuleType

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, fbeta_score
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper

from predictsignauxfaibles.config import IGNORE_NA
from predictsignauxfaibles.data import SFDataset


def merge_models(model_list: list):
    """
    Builds a single list of predicted probabilities based on several models,
    listed by decreasing order of priority.
    For a given SIRET, if no prediction is found for the first model in the list,
    a prediction for the next model will be considered, etc
    Arguments:
        model_list: list
            A list of pandas DataFrame containing, at least, the following columns: siren, predicted_probability
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
    if pred > t_rouge:
        return "Alerte seuil F1"
    elif pred > t_orange:
        return "Alerte seuil F2"
    return "Pas d'alerte"


def make_alert(preds: pd.DataFrame, t_rouge: float, t_orange: float):
    """
    Generates red/orange/green flags based on two thresholds
    """
    assert "predicted_probability" in preds.columns.tolist()
    preds["alert"] = preds["predicted_probability"].apply(
        lambda x: assign_flag(x, t_rouge, t_orange)
    )

    num_rouge = sum(preds["predicted_probability"] > t_rouge)
    num_orange = sum(preds["predicted_probability"] > t_orange)
    num_orange -= num_rouge
    print(f"{num_rouge} rouge ({round(num_rouge/preds.shape[0] * 100, 2)}%)")
    print(f"{num_orange} orange ({round(num_orange/preds.shape[0] * 100, 2)}%)")

    return preds


def make_precision_recall_curve(dataset: SFDataset, pp: Pipeline, conf: ModuleType):
    """
    pp must be a Pipeline object containing two steps:
    [0] a DataFrameMapper fitted on data similar to dataset.data
    [1] a model (for instance: sklearn.linear.LogisticRegression) fitted on data similar to dataset.data
    """
    dataset.replace_missing_data().remove_na(ignore=IGNORE_NA)
    dataset.data = run_pipeline(dataset.data, conf.TRANSFO_PIPELINE)

    X_raw = dataset.data[set(dataset.data.columns).difference(set(["outcome"]))]
    y = dataset.data["outcome"].astype(int).to_numpy()

    (_, mapper) = pp.steps[0]
    (_, model) = pp.steps[1]
    X = mapper.transform(X_raw)

    return precision_recall_curve(y, model.predict_proba(X)[:, 1])


def make_thresholds_from_fbeta(
    dataset: SFDataset,
    pp: Pipeline,
    conf: ModuleType,
    beta_F1: float = 0.5,
    beta_F2: float = 2,
):
    precision, recall, thresh = make_precision_recall_curve(dataset, pp, conf)

    f_beta_F1 = []
    f_beta_F2 = []
    for thr_id, thr in enumerate(thresh.tolist()):
        f_beta_F1.append(
            fbeta_score(
                y_true=test_outcomes,
                y_pred=(default_lr.predict_proba(test_features_mapped)[:, 1] >= thr),
                beta=beta_F1,
            )
        )
        f_beta_F2.append(
            fbeta_score(
                y_true=test_outcomes,
                y_pred=(default_lr.predict_proba(test_features_mapped)[:, 1] >= thr),
                beta=beta_F2,
            )
        )

    f_beta_F1 = np.array(f_beta_F1)
    f_beta_F2 = np.array(f_beta_F2)

    t_F1_id = np.argmax(f_beta_F1)
    t_F1 = thresh[t_F1_id]

    t_F2_id = np.argmax(f_beta_F2)
    t_F2 = thresh[t_F2_id]

    return (t_F1, t_F2)


def make_thresholds_from_conditions(
    dataset: SFDataset,
    pp: Pipeline,
    conf: ModuleType,
    min_precision_T1=0.75,
    min_recall_T2=0.75,
):
    precision, recall, thresh = make_precision_recall_curve(dataset, pp, conf)

    t_F1_id = np.argmax(precision >= min_precision_F1)
    t_F1 = thresh[t_F1_id]

    t_F2_id = np.argmax(recall[::-1][:-1] >= min_recall_F2)
    t_F2 = thresh[::-1][t_F2_id]

    return (t_F1, t_F2)
