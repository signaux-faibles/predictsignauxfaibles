# pylint: disable=invalid-name,too-many-arguments,too-many-function-args
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    fbeta_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from predictsignauxfaibles.data import SFDataset

# Mute logs from sklean_pandas
logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)


def make_precision_recall_curve(dataset: SFDataset, model_pipeline: Pipeline):
    """
    Preprocesses the data in dataset.data and computes the precision-recall curve
    from a trained model in a pipeline containing (mapper, model).
    Args:
        - dataset: a SFDataset object containing the data to be evaluated on
        - model_pipeline must be a Pipeline object containing two steps:
        [0] a DataFrameMapper fitted on data similar to dataset.data
        [1] a model (for instance: sklearn.linear.LogisticRegression)
            fitted on data similar to dataset.data
    """
    X_raw = dataset.data[set(dataset.data.columns).difference(set(["outcome"]))]
    y = dataset.data["outcome"].astype(int).to_numpy()

    (_, mapper) = model_pipeline.steps[0]
    (_, model) = model_pipeline.steps[1]
    X = mapper.transform(X_raw)

    return precision_recall_curve(y, model.predict_proba(X)[:, 1])


def make_thresholds_from_fbeta(
    features: pd.DataFrame,
    outcomes: np.array,
    model_pipeline: Pipeline,
    beta_F1: float = 0.5,
    beta_F2: float = 2,
    n_thr: int = 1000,
    thresh: np.array = None,
):
    """
    Finds the classification thresholds that maximise f_beta score.
    We choose to define both alert levels by the thresholds that maximise f_beta
    for a given beta. Typically, F1 alert threshold is tuned to favour precision,
    while F2 alert threshold favours recall.
    """
    if thresh is None:
        thresh = np.linspace(0, 1, n_thr)

    f_beta_F1 = []
    f_beta_F2 = []
    for thr in tqdm(thresh.tolist()):
        f_beta_F1.append(
            fbeta_score(
                y_true=outcomes,
                y_pred=(model_pipeline.predict_proba(features)[:, 1] >= thr),
                beta=beta_F1,
            )
        )
        f_beta_F2.append(
            fbeta_score(
                y_true=outcomes,
                y_pred=(model_pipeline.predict_proba(features)[:, 1] >= thr),
                beta=beta_F2,
            )
        )

    f_beta_F1 = np.array(f_beta_F1)
    f_beta_F2 = np.array(f_beta_F2)

    t_F1_id = np.argmax(f_beta_F1)
    t_F1 = thresh[t_F1_id]
    print(
        f"F1 - beta={beta_F1} - Optimal threshold: {t_F1} - f_{beta_F1}={np.max(f_beta_F1)}"
    )

    t_F2_id = np.argmax(f_beta_F2)
    t_F2 = thresh[t_F2_id]
    print(
        f"F2 - beta={beta_F2} - Optimal threshold: {t_F2} - f_{beta_F2}={np.max(f_beta_F2)}"
    )

    return (t_F1, t_F2)


def make_thresholds_from_conditions(
    precision: np.array,
    recall: np.array,
    thresh: np.array,
    min_precision_F1: float = 0.93,
    min_recall_F2: float = 0.63,
):
    """
    Finds the classification thresholds that maximise performance
    under conditions:
    - the precision of the F1 alert list must be greater than min_precision_F1
    - the recall of the F2 alert list (as a superset of the F1 alert list)
      must be greater than min_recall_F2
    """
    t_F1_id = np.argmax(precision >= min_precision_F1)
    t_F1 = thresh[t_F1_id]
    print(f"F1 - Precision>={min_precision_F1} - Optimal threshold: {t_F1}")

    t_F2_id = np.argmax(recall[::-1][:-1] >= min_recall_F2)
    t_F2 = thresh[::-1][t_F2_id]
    print(f"F2 - Recall>={min_recall_F2} - Optimal threshold: {t_F2}")

    return (t_F1, t_F2)


def evaluate(
    model: Pipeline,
    dataset: SFDataset,
    beta: float,
    thresh: float = 0.5,
):  # pylint: disable=too-many-locals
    """
    Returns evaluation metrics of model evaluated on df
    Args:
        model: a sklearn-like model with a predict method
        dataset: SFDataset containing the data to evaluate on
        beta: ponderation of the importance of recall relative to precision
        thresh:
            If provided, the model will classify an entry X as positive if predict_proba(X)>=thresh.
            Otherwise, the model classifies X as positive if predict(X)=1, ie predict_proba(X)>=0.5
    Dev note: To be turned into a SFModel method when refactoring models
    """
    y_true = dataset.data["outcome"]
    y_score = model.predict_proba(dataset.data)[:, 1]
    y_pred = y_score >= thresh

    aucpr = average_precision_score(
        y_true,
        y_score,
    )
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    (tn, fp, fn, tp) = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()
    fbeta = fbeta_score(
        y_true,
        y_pred,
        beta=beta,
    )
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        "aucpr": aucpr,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        f"f{beta}": fbeta,
        "precision": precision,
        "recall": recall,
    }
