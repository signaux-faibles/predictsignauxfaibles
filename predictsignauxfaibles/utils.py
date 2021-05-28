from datetime import datetime
import importlib.util
import logging
import math
from typing import NamedTuple, List
from pathlib import Path
import pytz

from predictsignauxfaibles.config import MODEL_FOLDER


class MongoDBQuery:
    """
    Helper class used to build MongoDB pipelines
    """

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
        """
        Adds a match stage to the pipeline.
        Args:
            date_min: first period to include, in the 'YYYY-MM-DD' format
            date_max: first period to exclude, in the 'YYYY-MM-DD' format
            min_effectif: the minimum number of employees a firm must have to be included
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
        """
        Adds a sort stage to the pipeline in order to always retrieve the same sample
        from the Features collection.
        """
        self.sort_stage = {"$sort": {"value.random_order": -1}}
        self.pipeline.append(self.sort_stage)
        return self

    def add_limit(self, limit: int):
        """
        Adds a limit stage to the pipeline.
        """
        self.limit_stage = {"$limit": limit}
        self.pipeline.append(self.limit_stage)
        return self

    def add_replace_root(self):
        """
        Adds a replace root stage to the pipeline.
        """
        self.replace_root_stage = {"$replaceRoot": {"newRoot": "$value"}}
        self.pipeline.append(self.replace_root_stage)
        return self

    def add_projection(self, fields: List):
        """
        Adds a projection stage to filter only the fields in which we are interested.
        """
        self.project_stage = {"$project": {field: 1 for field in fields}}
        self.pipeline.append(self.project_stage)
        return self

    def to_pipeline(self) -> List:
        """
        Returns the pipeline.
        """
        return self.pipeline

    def reset(self):
        """
        Empties the pipeline.
        """
        self.pipeline = []

    @staticmethod
    def __date_to_iso(date: str):
        """
        Converts a date in the YYYY-MM-DD format into a
        datetime usable by mongodb"
        """
        return pytz.utc.localize(datetime.strptime(date, "%Y-%m-%d"))


class CLIError(Exception):
    "Base class for errors linked to reading CLI options"


class EmptyFileError(CLIError):
    "Raised when reading an empty file that should be non-empty"


def check_feature(feature_name: str, variables: list, pipeline: List[NamedTuple]):
    """
    Check that a feature is either explicitly requested from the database as a variable
    or is created by a step in PIPELINE.
    """
    is_ok = False
    if feature_name in variables:
        is_ok = True
    for step in pipeline:
        if step.output is not None and feature_name in step.output:
            is_ok = True
    return is_ok


def set_if_not_none(obj, attr, val):
    """
    Sets the attribute of an object with some value if that value is not null
    Args:
        obj: any object
        attr: the attribute to be set
        val: the value to set. If None, attr will remain unchanged
    """
    if val is not None:
        setattr(obj, attr, val)


def sigmoid(flt: float):
    """Returns the sigmoid of flt"""
    return 1 / (1 + math.exp(-flt))


def load_conf(model_name: str = "default"):
    """
    Loads a model configuration from a model name
    Args:
        model_name: str
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
