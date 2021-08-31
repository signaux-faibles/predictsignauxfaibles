import importlib.util
import logging
import math
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import OneHotEncoder

from predictsignauxfaibles.config import MODEL_FOLDER


class MongoDBQuery:
    """Helper class used to build MongoDB pipelines."""

    def __init__(self):
        self.pipeline = []
        self.match_stage = None
        self.sort_stage = None
        self.limit_stage = None
        self.replace_root_stage = None
        self.project_stage = None

    def add_standard_match(  # pylint: disable=too-many-arguments
        self,
        date_min: str,
        date_max: str,
        min_effectif: int,
        sirets: List = None,
        sirens: List = None,
        categorical_filters: dict = None,
        outcome: List = None,
    ):
        """Adds a match stage to the pipeline.

        Args:
            date_min: First period to include, in the 'YYYY-MM-DD' format.
            date_max: First period to exclude, in the 'YYYY-MM-DD' format
            min_effectif: The minimum number of employees a firm must have to be
              included.

        """
        self.match_stage = {
            "$match": {
                "$and": [
                    {
                        "value.random_order": {"$gte": 0}
                    },  # this forces Mongo to use the Index
                    {
                        "_id.periode": {
                            "$gte": self.__date_to_iso(date_min),
                            "$lt": self.__date_to_iso(date_max),
                        }
                    },
                    {"value.effectif": {"$gte": min_effectif}},
                ]
            }
        }

        if sirets is not None:
            self.match_stage["$match"]["$and"].append({"_id.siret": {"$in": sirets}})

        if sirens is not None:
            self.match_stage["$match"]["$and"].append({"value.siren": {"$in": sirens}})

        if categorical_filters is not None:
            for (category, cat_filter) in categorical_filters.items():
                if isinstance(cat_filter, (list, tuple)):
                    self.match_stage["$match"]["$and"].append(
                        {f"value.{category}": {"$in": cat_filter}}
                    )
                elif isinstance(cat_filter, (int, float, str)):
                    self.match_stage["$match"]["$and"].append(
                        {f"value.{category}": cat_filter}
                    )
                else:
                    logging.warning(
                        f"Ignored filter of unknown type on category {category}."
                    )
                    continue

        if outcome is not None:
            self.match_stage["$match"]["$and"].append({"value.outcome": outcome})
        else:
            self.match_stage["$match"]["$and"].append(
                {"value.outcome": {"$in": [True, False]}}
            )

        self.pipeline.append(self.match_stage)

        return self

    def add_sort(self):
        """Adds a sort stage to the pipeline.

        This allows to always retrieve the same sample from the Features collection.
        """
        self.sort_stage = {"$sort": {"value.random_order": -1}}
        self.pipeline.append(self.sort_stage)
        return self

    def add_limit(self, limit: int):
        """Adds a limit stage to the pipeline."""
        self.limit_stage = {"$limit": limit}
        self.pipeline.append(self.limit_stage)
        return self

    def add_replace_root(self):
        """Adds a replace root stage to the pipeline."""
        self.replace_root_stage = {"$replaceRoot": {"newRoot": "$value"}}
        self.pipeline.append(self.replace_root_stage)
        return self

    def add_projection(self, fields: List):
        """Adds a projection stage.

        This allows to filter only the fields in which we are interested."""
        self.project_stage = {"$project": {field: 1 for field in fields}}
        self.pipeline.append(self.project_stage)
        return self

    def to_pipeline(self) -> List:
        """Returns the pipeline."""
        return self.pipeline

    def reset(self):
        """Empties the pipeline."""
        self.pipeline = []

    @staticmethod
    def __date_to_iso(date: str):
        """Converts a date from YYYY-MM-DD format into a datetime usable by mongodb."""
        return pytz.utc.localize(datetime.strptime(date, "%Y-%m-%d"))


class CLIError(Exception):
    "Base class for errors linked to reading CLI options."


class EmptyFileError(CLIError):
    "Raised when reading an empty file that should be non-empty."


def check_feature(
    feature_name: str, variables: list, pipeline: List[NamedTuple]
) -> bool:
    """Sanity checks a feature.

    Check that a feature is either explicitly requested from the database as a variable
    or is created by a step in PIPELINE.

    Args:
        feature_name: The feature's name
        variables: A list of variable namesq
        pipeline: A processing pipeline.

    Returns:
        If the feature passed the tests.
    """
    is_ok = False
    if feature_name in variables:
        is_ok = True
    for step in pipeline:
        if step.output is not None and feature_name in step.output:
            is_ok = True
    return is_ok


def set_if_not_none(obj, attr, val):
    """Sets the attribute of an object with some value if that value is not null.

    Args:
        obj: Any object.
        attr: The attribute to be set.
        val: The value to set. If None, `attr` will remain unchanged.

    """
    if val is not None:
        setattr(obj, attr, val)


def sigmoid(flt: float) -> float:
    """Returns the value of a logistic function evaluated at `flt`."""
    return 1 / (1 + math.exp(-flt))


def load_conf(model_name: str = "default") -> ModuleType:
    """Loads a model configuration from a model name.

    Args:
        model_name: str

    Returns:
        The model configuration, as a module.

    """
    conf_filepath = Path(MODEL_FOLDER) / model_name / "model_conf.py"
    if not conf_filepath.exists():
        raise ValueError(f"{conf_filepath} does not exist")

    spec = importlib.util.spec_from_file_location(
        f"models.{model_name}.model_conf", conf_filepath
    )
    model_conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_conf)  # pylint: disable=wrong-import-position
    return model_conf


def assign_flag(pred: float, t_rouge: float, t_orange: float) -> str:
    """Turns a prediction score pred into an alert level.

    Args:
        pred: A predicted risk score between 0 and 1
        t_rouge: A threshold, between 0 and 1, used for selecting "high risk" records.
        t_orange: A threshold, between 0 and t_rouge, used for selecting "moderate
          risk" records. `t_orange` should not be greater than `t_rouge`.
          If `t_orange == t_rouge`, there will indeed be a single risk level (high
          risk).


    """
    assert 0 <= t_rouge <= 1, "t_rouge must be a number between 0 and 1"
    assert 0 <= t_orange <= 1, "t_orange must be a number between 0 and 1"
    assert t_orange <= t_rouge, "t_rouge should be greater than t_orange"

    if pred > t_rouge:
        return "Alerte seuil F1"
    if pred > t_orange:
        return "Alerte seuil F2"
    return "Pas d'alerte"


def log_splits_size(
    predicted_proba: np.array, thresh_f1: float, thresh_f2: float
):  # pylint: disable=line-too-long
    """Generates red/orange/green flags based on two thresholds.'

    Args:
        predicted_proba: An array of predictions.
        thresh_f1: The first threshold.
        thresh_f2: The second threshold.
    """
    num_f1 = sum(predicted_proba > thresh_f1)
    num_f2 = sum(predicted_proba > thresh_f2)
    num_f2 -= num_f1
    logging.info(
        f'Pallier "risque fort" (rouge): {num_f1} ({round(num_f1/len(predicted_proba) * 100, 2)}%)'
    )
    logging.info(
        f'Pallier "risque modéré" (orange): {num_f2} ({round(num_f2/len(predicted_proba) * 100, 2)}%)'
    )


def map_cat_feature_to_categories(data: pd.DataFrame, conf: ModuleType):
    """Maps categorical features to variables.

    The categories are built from feature levels **in the same order that they are
    found in** `conf.FEATURE_GROUPS` using a OneHotEncoder fitted to each categorical
    variable.

    Args:
        data: The features data.
        conf: A configuration module.

    Returns:

    """
    group_feats_tuples = [
        (group, feat)
        for (group, feats) in conf.FEATURE_GROUPS.items()
        for feat in feats
    ]
    data = data[conf.FEATURES]
    data.columns = group_feats_tuples

    cat_mapping = {}
    for (group, feats) in conf.FEATURE_GROUPS.items():
        for feat in feats:
            if feat not in conf.TO_ONEHOT_ENCODE:
                continue
            feat_oh = OneHotEncoder()
            feat_oh.fit(
                data[
                    [
                        (group, feat),
                    ]
                ]
            )
            cat_names = feat_oh.get_feature_names().tolist()
            cat_mapping[(group, feat)] = [feat + "_" + name for name in cat_names]

    return cat_mapping


def make_multi_columns(data: pd.DataFrame, conf: ModuleType) -> List[Tuple]:
    """Builds a multicolumn-compatible list of tuples.

    This list corresponds to a list of (group, feature), where:
    - `group` is the name of a group of features (by source or subject),
    - `feature` is a variable that can be used in a model

    Categorical variables are exploded into this list of tuples.

    Args:
        data: Data containing categorical features.
        conf: The associated configuration file.

    Returns:
        The list of (group, feat) tuples.
    """
    # Mapping categorical variables to their oh-encoded level variables.
    cat_mapping = map_cat_feature_to_categories(data, conf)

    # Create a list of tuples than can be turned into a MultiIndex.
    multi_columns_cat = []
    for (group, feats) in conf.FEATURE_GROUPS.items():
        for feat in feats:
            if (group, feat) in cat_mapping.keys():
                for cat_feat in cat_mapping[(group, feat)]:
                    multi_columns_cat.append((group, cat_feat))

    multi_columns_nocat = [
        (group, feat)
        for (group, feats) in conf.FEATURE_GROUPS.items()
        for feat in feats
        if (group, feat) not in cat_mapping.keys()
    ]

    return multi_columns_cat + multi_columns_nocat
