import logging
from typing import List

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

import predictsignauxfaibles.config as config
from predictsignauxfaibles.utils import MongoDBQuery, parse_yml_config
from predictsignauxfaibles.decorators import is_random


class SFDataset:
    """
    Retrieve a signaux faibles dataset.
    Args:
        date_min: first period to include, in the 'YYYY-MM-DD' format. Default is first.
        date_max: first period to exclude, in the 'YYYY-MM-DD' format Default is latest.
        fields: which fields of the Features collection to retrieve. Default is all.
        sample_size: max number of (siret x period) rows to retrieve. Default is all.
        sirets: a list of SIRET to select.
        outcome: restrict query to firms that fall in a specific outcome (True / False)
        min_effectif: min number of employees for firm to be in the sample (defaults to your .env)

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        date_min: str = "1970-01-01",
        date_max: str = "3000-01-01",
        fields: List = None,
        sample_size: int = 0,  # a sample size of 0 means all data is retrieved
        min_effectif: int = "default",
        sirets: List = None,
        sirens: List = None,
        outcome: bool = None,
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
        self.min_effectif = (
            config.MIN_EFFECTIF if min_effectif == "default" else min_effectif
        )
        self.sirets = sirets
        self.sirens = sirens
        self.outcome = outcome
        self.mongo_pipeline = MongoDBQuery()

    def _connect_to_mongo(self):
        if self.__mongo_client is None:
            logging.debug("opening connection")
            self.__mongo_client = MongoClient(host=config.MONGODB_PARAMS.url)
            self.__mongo_database = self.__mongo_client.get_database(
                config.MONGODB_PARAMS.db
            )
            self.__mongo_collection = self.__mongo_database.get_collection(
                config.MONGODB_PARAMS.collection
            )

    def _disconnect_from_mongo(self):
        if self.__mongo_client is not None:
            logging.debug("closing connection")
            self.__mongo_client.close()
            self.__mongo_client = None
            self.__mongo_database = None
            self.__mongo_collection = None

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
        )

    def fetch_data(self, warn: bool = True):
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        Args:
            warn : emmit a warning if fetch_data is overwriting some already existing data
        """

        self.mongo_pipeline.reset()

        if warn and self.data is not None:
            logging.warning("Dataset object was not empty. Overriding...")

        self.mongo_pipeline.add_standard_match(
            self.date_min,
            self.date_max,
            self.min_effectif,
            sirets=self.sirets,
            sirens=self.sirens,
            outcome=self.outcome,
        )
        self.mongo_pipeline.add_sort()
        self.mongo_pipeline.add_limit(self.sample_size)
        self.mongo_pipeline.add_replace_root()

        if self.fields is not None:
            self.mongo_pipeline.add_projection(self.fields)

        self._connect_to_mongo()
        cursor = self.__mongo_collection.aggregate(self.mongo_pipeline.to_pipeline())

        self.data = self.__cursor_to_df(cursor)
        self._disconnect_from_mongo()

        return self

    def prepare_data(
        self,
        remove_strong_signals: bool = False,
        defaults_map: dict = None,
        cols_ignore_na: list = None,
    ):
        """
        Run data preparation operations on the dataset.
        remove_strong_signals drops observations with time_til_outcome <= 0
         (i.e. firms that are already in default).
        """
        assert isinstance(
            self.data, pd.DataFrame
        ), "DataFrame not found. Please fetch data first."

        if defaults_map is None:
            defaults_map = config.DEFAULT_DATA_VALUES

        if cols_ignore_na is None:
            cols_ignore_na = config.IGNORE_NA

        logging.info("Replacing missing data with default values")
        self._replace_missing_data(defaults_map)

        logging.info("Drop observations with missing required fields.")
        self._remove_na(cols_ignore_na)

        if remove_strong_signals:
            logging.info("Removing 'strong signals'.")
            self._remove_strong_signals()

        logging.info("Resetting index for DataFrame.")
        self.data.reset_index(drop=True, inplace=True)
        return self

    def _remove_strong_signals(self):
        """
        Strong signals is when a firm is already in default (time_til_outcome <= 0)
        """
        assert (
            "time_til_outcome" in self.data.columns
        ), "The `time_til_outcome` column is needed in order to remove strong signals."

        self.data = self.data[~(self.data["time_til_outcome"] <= 0)]

    def _replace_missing_data(self, defaults_map: dict = None):
        """
        Replace missing data with defaults defined in project config
        Args:
            defaults_map: a dictionnary in the {column_name: default_value} format
        """
        if defaults_map is None:
            defaults_map = config.DEFAULT_DATA_VALUES

        for feature, default_value in defaults_map.items():
            logging.debug(f"Column {feature} defaulting to value {default_value}")

        for column in defaults_map:
            try:
                self.data[column] = self.data[column].fillna(defaults_map.get(column))
            except KeyError:
                logging.debug(f"Column {column} not in dataset")
                continue

    def _remove_na(self, cols_ignore_na: list):
        """
        Remove all observations with missing values.
        """

        cols_drop_na = set(self.data.columns).difference(set(cols_ignore_na))

        logging.info("Removing NAs from dataset.")
        for feature in cols_drop_na:
            logging.debug(
                f"Rows with NAs in field {feature} will be dropped, unless default val is provided"
            )

        for feature in cols_ignore_na:
            logging.debug(f"Rows with NAs in field {feature} will NOT be dropped")

        logging.info(f"Number of observations before: {len(self.data.index)}")
        self.data.dropna(subset=cols_drop_na, inplace=True)
        logging.info(f"Number of observations after: {len(self.data.index)}")

    def __repr__(self):
        out = f"""
Signaux Faibles Dataset
----------------------------------------------------
{self.data.head() if isinstance(self.data, pd.DataFrame) else "Empty Dataset"}
[...]
----------------------------------------------------
Number of observations = {len(self) if isinstance(self.data, pd.DataFrame) else "0"}
        """
        return out

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        """
        Length of SFDataset is the length of its dataframe
        """
        return len(self.data) if isinstance(self.data, pd.DataFrame) else 0

    @staticmethod
    def __cursor_to_df(cursor: Cursor):
        """
        Extract data from a MongoDB cursor into a Pandas dataframe
        """
        return pd.DataFrame(cursor)


class OversampledSFDataset(SFDataset):
    """
    Helper class to oversample a SFDataset
    Args:
        proportion_positive_class: the desired proportion of firms for which outcome = True
    """

    def __init__(self, proportion_positive_class: float, **kwargs):
        super().__init__(**kwargs)
        assert (
            0 <= proportion_positive_class <= 1
        ), "proportion_positive_class must be between 0 and 1"
        self.proportion_positive_class = proportion_positive_class

    @is_random
    def fetch_data(self):  # pylint: disable=arguments-differ
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Store the resulting data in the `data` attribute
        """
        # compute the number of lines to fetch with outcome = True
        n_obs_true = round(self.proportion_positive_class * self.sample_size)
        n_obs_false = self.sample_size - n_obs_true

        # fetch true
        self.sample_size = n_obs_true
        self.outcome = True
        super().fetch_data(warn=False)
        true_data = self.data

        # fetch false
        self.sample_size = n_obs_false
        self.outcome = False
        super().fetch_data(warn=False)
        false_data = self.data
        full_data = true_data.append(false_data)
        self.data = full_data.sample(frac=1).reset_index(drop=True)
