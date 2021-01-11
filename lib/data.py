import logging
from typing import List

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

import config
from lib.utils import MongoDBQuery


class SFDataset:
    """
    Retrieve a signaux faibles dataset.
    The batch_id argument is optional and by default is the one contained in your .env file.
    """

    def __init__(self, batch_id: str = "default"):
        self.__mongo_client = MongoClient(host=config.MONGODB_PARAMS.url)
        self.__mongo_database = self.__mongo_client.get_database(
            config.MONGODB_PARAMS.db
        )
        self.__mongo_collection = self.__mongo_database.get_collection(
            config.MONGODB_PARAMS.collection
        )
        self.data = None
        self.batch_id = config.BATCH_ID if batch_id == "default" else batch_id
        self.mongo_pipeline = MongoDBQuery()

    def fetch_data(
        self,
        date_min: str = "1970-01-01",
        date_max: str = "3000-01-01",
        fields: List = None,
        sample_size: int = 0,  # a sample size of 0 means all data is retrieved
        **kwargs,
    ):
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        Args:
            date_min: first period to include, in the 'YYYY-MM-DD' format. Default is first.
            date_max: first period to exclude, in the 'YYYY-MM-DD' format Default is latest.
            fields: which fields of the Features collection to retrieve. Default is all.
            sample_size: max number of (siret x period) rows to retrieve. Default is all.

        Additionally, the following parameters are recognized:
            min_effectif: the minimum number of employees a firm must have to be in the sample.
            sirets: a list of SIRET to select.
        """
        if self.data is not None:
            logging.warning("Dataset object was not empty. Overriding...")

        min_effectif = kwargs.get("min_effectif", config.MIN_EFFECTIF)
        sirets = kwargs.get("sirets")
        self.mongo_pipeline.add_standard_match(
            date_min, date_max, min_effectif, self.batch_id, sirets=sirets
        )
        self.mongo_pipeline.add_sort()
        self.mongo_pipeline.add_limit(sample_size)
        self.mongo_pipeline.add_replace_root()

        if fields is not None:
            self.mongo_pipeline.add_projection(fields)

        cursor = self.__mongo_collection.aggregate(self.mongo_pipeline.to_pipeline())

        self.data = self.__cursor_to_df(cursor)
        return self

    def prepare_data(self):
        """
        Run data preparation operations on the dataset.
        """
        assert isinstance(
            self.data, pd.DataFrame
        ), "DataFrame not found. Please fetch data first."

        logging.info("Creating a `siren` column")
        self._add_siren()

        logging.info("Replacing missing data with default values")
        self._replace_missing_data(defaults_map=config.DEFAULT_DATA_VALUES)

        logging.info("Drop observations with missing required fields.")
        self._remove_na()

    def _add_siren(self):
        """
        Add a `siren` column to the dataset.
        """
        assert "siret" in self.data.columns, "siret column not found"
        self.data["siren"] = self.data["siret"].apply(lambda siret: siret[:9])

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

    def _summarize_dataset(self):
        """
        Returns a summary of the current state of the dataset.
        """
        summary = {
            "batch_id": self.batch_id,
            "pipeline": self.mongo_pipeline.to_pipeline(),
            "has_data": False,
        }

        if isinstance(self.data, pd.DataFrame):
            summary["has_data"] = True
            summary["fields"] = self.data.columns

        return summary

    def __repr__(self):
        summary = self._summarize_dataset()
        out = f"""
        -----------------------
        Signaux Faibles Dataset
        -----------------------

        batch_id : {summary["batch_id"]}
        ---------- 

        Fields:
        -------
            {summary["fields"]}

        MongoDB Aggregate Pipeline:
        ---------------------------
            {summary["pipeline"]}
        """

        return out

    @staticmethod
    def __cursor_to_df(cursor: Cursor):
        """
        Extract data from a MongoDB cursor into a Pandas dataframe
        """
        return pd.DataFrame(cursor)
