import logging
import math
import re
from typing import List, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from predictsignauxfaibles.data import SFDataset


class SqrtTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    
    def fit(self, X, y = None):
        return self
    
    
    def transform(self, X, y = None):
        X_ = X.copy()
        X_ = np.sqrt(X_)
        return X_


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    
    def fit(self, X, y = None):
        return self
    
    
    def transform(self, X, y = None):
        X_ = X.copy()
        X_ = np.log(X_+1)
        return X_

    
def apply_log(number: float) -> float:
    """
    Apply the transformation number -> log(number + 1)
    """
    return math.log(number + 1)


def apply_sqrt(number: float) -> float:
    """
    Apply the transformation number -> sqrt(number)
    """
    return math.sqrt(number)

    
def print_featuring_for_model_conf(featuring: pd.DataFrame) -> str:
    """
    Print the tranformation for each feature to be plugged in a model_conf.py.
    """
    featuring = featuring[['func', 'feature']].groupby('func').agg(list).reset_index()
    featuring = {func:feat for (feat, func) in zip(featuring['feature'], featuring['func'])}
    return featuring


def get_featuring(
    features: List[str], funcs: List[Callable[[float], float]]
) -> pd.DataFrame:
    """
    For each feature in 'features' taken separately, provide the transformation among
    'funcs' to apply to improve the prediction performance with a LogisticRegression
    (outcome = feature). For significant results, this should be bootstrap but it
    is enough to test the transformations in the model used in the project in a
    second step.
    """
    dataset = SFDataset(
        date_min="2015-01-01",
        date_max="2020-06-30",
        fields=["outcome"] + features,
        sample_size=1000,
    )
    dataset.fetch_data()

    res = [
        get_featuring_unitary(dataset.data, feat, f) for feat in features for f in funcs
    ]

    res_as_df = pd.DataFrame(res)
    res_as_df = res_as_df[res_as_df["is_relevant"]]

    return res_as_df.sort_values("score_after", ascending=False).drop_duplicates(
        ["feature"]
    )


def get_featuring_unitary(
    data: pd.DataFrame, feat: str, func: Callable[[float], float]
) -> dict:
    """
    Apply the transformation 'func' to the feature 'feat', build a LogisticRegression
    (outcome = feature) with and without the transformation and determine if it was
    relevant.
    """
    data = data[["outcome", feat]].copy()

    # handle missing value
    data.dropna(inplace=True)

    if len(data[feat]) == 0:
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    # handle non-numeric
    if not all([type(x) in [int, float] for x in data[feat]]):
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    # handle negative values
    if any(data[feat] < 0):
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }

    response = data.outcome

    # handle singular class in the response
    if len(response.unique()) == 1:
        return {
            "feature": feat,
            "func": func.__name__,
            "score_before": np.nan,
            "score_after": np.nan,
            "is_relevant": False,
        }
    feat_values = np.array(data[feat]).reshape(-1, 1)

    # Logistic without featuring
    model = LogisticRegression()
    model.fit(feat_values, response)
    score_before = model.score(feat_values, response)

    # Logistic with featuring
    feat_values = np.array(list(map(func, feat_values))).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(feat_values, response)
    score_after = model.score(feat_values, response)

    return {
        "feature": feat,
        "func": func.__name__,
        "score_before": score_before,
        "score_after": score_after,
        "is_relevant": score_after > score_before,
    }
