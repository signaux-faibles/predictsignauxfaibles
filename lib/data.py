import logging
from typing import List

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

from config import MONGODB_PARAMS, MIN_EFFECTIF, BATCH_ID
from lib.utils import MongoDBQuery


class SFDataset:
    """
    Retrieve a signaux faibles dataset
    """

    def __init__(self, batch_id: str = "default"):
        self.__mongo_client = MongoClient(host=MONGODB_PARAMS.url)
        self.__mongo_database = self.__mongo_client.get_database(MONGODB_PARAMS.db)
        self.__mongo_collection = self.__mongo_database.get_collection(
            MONGODB_PARAMS.collection
        )
        self.data = None
        self.batch_id = BATCH_ID if batch_id == "default" else batch_id
        self.mongo_pipeline = MongoDBQuery()

    def get_data(
        self,
        date_min: str = "1970-01-01",
        date_max: str = "3000-01-01",
        fields: List = None,
        sample_size: int = 0,  # a sample size of 0 means all data is retrieved
        **kwargs,
    ):
        """
        Retrieve query from MongoDB database using the Aggregate framework
        Args:
            date_min: first period to include, in the 'YYYY-MM-DD' format. Default is first.
            date_max: first period to exclude, in the 'YYYY-MM-DD' format Default is latest.
            fields: which fields of the Features collection to retrieve. Default is all.
            sample_size: max number of (siret x period) rows to retrieve. Default is all.

        Additionally, the following parameters are recognized:
            min_effectif: the minimum number of employees a firm must have to be in the sample.
            sirets: a list of SIRET to select.
            find_query: override all previous parameters and evaluate a custom MongoDB find query.
        """
        if self.data is not None:
            logging.warning("Dataset object was not empty. Overriding...")

        if "find_query" in kwargs:
            find_query = kwargs["find_query"]
            logging.info(f"Using custom query {find_query}")
            cursor = self.__mongo_collection.find(find_query, {"_id": False}).limit(
                sample_size
            )
        else:
            min_effectif = kwargs.get("min_effectif", MIN_EFFECTIF)
            sirets = kwargs.get("sirets")
            self.mongo_pipeline.add_standard_match(
                date_min, date_max, min_effectif, self.batch_id, sirets=sirets
            )
            self.mongo_pipeline.add_sort()
            self.mongo_pipeline.add_limit(sample_size)
            self.mongo_pipeline.add_replace_root()

            if fields is not None:
                self.mongo_pipeline.add_projection(fields)

            cursor = self.__mongo_collection.aggregate(
                self.mongo_pipeline.to_pipeline()
            )

        self.data = self.__cursor_to_df(cursor)
        return self

    def summarize_dataset(self):
        """
        Returns a summary of the dataset.
        """
        summary = {
            "batch_id": self.batch_id,
            "pipeline": self.mongo_pipeline.to_pipeline(),
            "has_data": False,
        }

        if self.data is not None:
            summary["has_data"] = True
            summary["fields"] = self.data.columns

        return summary

    def __repr__(self):
        summary = self.summarize_dataset()
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
        # return pd.DataFrame([element["value"] for element in cursor])
        return pd.DataFrame(cursor)
