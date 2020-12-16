import logging

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

from config import MONGODB_PARAMS
from lib.utils import MongoDBQuery


class SFDataset:
    """
    Retrieve a signaux faibles dataset
    """

    def __init__(self):
        self.__mongo_client = MongoClient(host=MONGODB_PARAMS.url)
        self.__mongo_database = self.__mongo_client.get_database(MONGODB_PARAMS.db)
        self.__mongo_collection = self.__mongo_database.get_collection(
            MONGODB_PARAMS.collection
        )
        self.data = None

    def get_data(
        self,
        # date_min: str = None,
        # date_max: str = None,
        # sample_size: int = None,
        # batch: str = "latest",
        find_query: str = None,
        limit: int = 0,  # no limit
    ):
        """
        Retrieve query from MongoDB database


        sample_size: max number of (siret x period) rows to retrieve

        """
        if self.data:
            logging.warning("Dataset object was not empty. Overriding...")

        if find_query:
            cursor = self.__mongo_collection.find(find_query, {"_id": False}).limit(
                limit
            )
        else:
            mongo_pipeline = MongoDBQuery().add_limit(1).to_pipeline()  # TODO
            cursor = self.__mongo_collection.aggregate(mongo_pipeline)

        self.data = self.__cursor_to_df(cursor)
        return self

    def list_available_fields(self):
        """
        Return a list of data fields available in the collection
        """
        # TODO implement this if possible (mongodb is schema-less)
        return self.__mongo_collection.find_one()

    @staticmethod
    def __cursor_to_df(cursor: Cursor):
        """
        Extract data from a MongoDB cursor into a Pandas dataframe
        """
        return pd.DataFrame([element["value"] for element in cursor])
