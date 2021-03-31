import logging
from pathlib import Path
from typing import NamedTuple, List

from datetime import datetime
import jsonschema
import pytz
import yaml


class MongoParams(NamedTuple):
    """
    MongoDb parameters used in config
    """

    url: str
    db: str
    collection: str


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


def parse_yml_config(path: str):
    """
    Converts a YAML config file into a python object
    Args:
        path: a string path to the yaml file
    """
    parsed_path = Path(path)
    if not parsed_path.exists():
        raise ConfigFileError(f"path {parsed_path} does not exist")
    if not parsed_path.is_file():
        raise ConfigFileError(f"{parsed_path} is not a file")
    if parsed_path.suffix != ".yml":
        raise ConfigFileError("config file must be a .yml file")
    data = yaml.load(parsed_path.read_bytes(), Loader=yaml.Loader)
    jsonschema.validate(data, CONFIG_FILE_SCHEMA)
    return data


# JSONschema schema used to validate config files
CONFIG_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "target": {"type": "string"},
        "features": {"type": "array"},
        "train_on": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "sample_size": {"type": "integer"},
            },
            "required": ["start_date", "end_date", "sample_size"],
        },
        "predict_on": {"type": "string"},
    },
    "required": [
        "name",
        "version",
        "target",
        "features",
        "train_on",
        "predict_on",
    ],
}


class ConfigFileError(Exception):
    """
    Error class for Configuration Files-related issues
    """
