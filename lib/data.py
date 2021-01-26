import logging
from typing import List

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

import config
from lib.utils import MongoDBQuery, parse_yml_config


class SFDataset:
    """
    Retrieve a signaux faibles dataset.
    Args:
        date_min: first period to include, in the 'YYYY-MM-DD' format. Default is first.
        date_max: first period to exclude, in the 'YYYY-MM-DD' format Default is latest.
        fields: which fields of the Features collection to retrieve. Default is all.
        sample_size: max number of (siret x period) rows to retrieve. Default is all.
        sirets: a list of SIRET to select.
        batch_id : MongoDB batch id (defaults to your .env)
        min_effectif: min number of employees for firm to be in the sample (defaults to your .env)

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        date_min: str = "1970-01-01",
        date_max: str = "3000-01-01",
        fields: List = None,
        sample_size: int = 0,  # a sample size of 0 means all data is retrieved
        batch_id: str = "default",
        min_effectif: int = "default",
        sirets: List = None,
    ):
        self.__mongo_client = MongoClient(host=config.MONGODB_PARAMS.url)
        self.__mongo_database = self.__mongo_client.get_database(
            config.MONGODB_PARAMS.db
        )
        self.__mongo_collection = self.__mongo_database.get_collection(
            config.MONGODB_PARAMS.collection
        )
        self.data = None
        self.date_min = date_min
        self.date_max = date_max
        self.fields = fields
        self.sample_size = sample_size
        self.batch_id = config.BATCH_ID if batch_id == "default" else batch_id
        self.min_effectif = (
            config.MIN_EFFECTIF if min_effectif == "default" else min_effectif
        )
        self.sirets = sirets
        self.mongo_pipeline = MongoDBQuery()

    @classmethod
    def from_config_file(cls, path: str, mode: str = "train"):
        """
        Instantiate a SFDataset object via a yaml config file
        Args:
            path: path to config file (typically in ./models/{version}/model.yml)
            mode: "train" or "predict". Whether the dataset is for training or for predicting.
        """
        conf = parse_yml_config(path)
        if mode not in {"train", "predict"}:
            raise ValueError("'mode' must be one of 'train' or 'predict")
        return cls(
            date_min=conf[f"{mode}_on"]["start_date"],
            date_max=conf[f"{mode}_on"]["end_date"],
            fields=conf["features"] + [conf["target"]] + ["siret", "periode"],
            sample_size=conf[f"{mode}_on"].get("sample_size", 0),
            batch_id=conf["batch_id"],
        )

    def fetch_data(self):
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        """

        self.mongo_pipeline.reset()

        if self.data is not None:
            logging.warning("Dataset object was not empty. Overriding...")

        self.mongo_pipeline.add_standard_match(
            self.date_min,
            self.date_max,
            self.min_effectif,
            self.batch_id,
            sirets=self.sirets,
        )
        self.mongo_pipeline.add_sort()
        self.mongo_pipeline.add_limit(self.sample_size)
        self.mongo_pipeline.add_replace_root()

        if self.fields is not None:
            self.mongo_pipeline.add_projection(self.fields)

        cursor = self.__mongo_collection.aggregate(self.mongo_pipeline.to_pipeline())

        self.data = self.__cursor_to_df(cursor)
        return self

    def prepare_data(self, remove_strong_signals: bool = False):
        """
        Run data preparation operations on the dataset.
        remove_strong_signals drops observations with time_til_outcome <= 0
         (i.e. firms that are already in default).
        """
        assert isinstance(
            self.data, pd.DataFrame
        ), "DataFrame not found. Please fetch data first."

        logging.info("Replacing missing data with default values")
        self._replace_missing_data(defaults_map=config.DEFAULT_DATA_VALUES)

        logging.info("Drop observations with missing required fields.")
        self._remove_na()

        if remove_strong_signals:
            self._remove_strong_signals()

        logging.info("Resetting index for DataFrame.")
        self.data.reset_index(drop=True, inplace=True)

    def _remove_strong_signals(self):
        assert (
            "time_til_outcome" in self.data.columns
        ), "The `time_til_outcome` column is needed in order to remove strong signals."

        self.data = self.data.loc[
            self.data["time_til_outcome"].isna() | self.data["time_til_outcome"] > 0
        ]

    def _replace_missing_data(self, defaults_map: dict):
        """
        Replace missing data with defaults defined in project config
        Args:
            defaults_map: a dictionnary in the {column_name: default_value} format
        """
        for column in defaults_map:
            try:
                self.data[column] = self.data[column].fillna(defaults_map.get(column))
            except KeyError:
                logging.debug(f"Column {column} not in dataset")
                continue

    def _remove_na(self):
        """
        Remove all observations with missing values.
        """
        logging.info("Removing NAs from dataset.")
        logging.info(f"Number of observations before: {len(self.data.index)}")
        self.data.dropna(inplace=True)
        logging.info(f"Number of observations after: {len(self.data.index)}")

    def __repr__(self):
        out = f"""
        -----------------------
        Signaux Faibles Dataset
        -----------------------

        batch_id : {self.batch_id}
        ---------- 

        Fields:
        -------
            {self.fields if len(self.fields)>1 else "all"}

        MongoDB Aggregate Pipeline:
        ---------------------------
            {self.mongo_pipeline.to_pipeline()}
        """

        return out

    @staticmethod
    def __cursor_to_df(cursor: Cursor):
        """
        Extract data from a MongoDB cursor into a Pandas dataframe
        """
        return pd.DataFrame(cursor)
