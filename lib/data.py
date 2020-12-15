import logging

import pandas as pd
from pymongo import MongoClient
from pymongo.cursor import Cursor

from config import MONGODB_PARAMS


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

    def get_data(self, query: str):
        """
        Retrieve query from MongoDB database
        """
        if self.data:
            logging.warning("Dataset object was not empty. Overriding...")
        cursor = self.__mongo_collection.find(query, {"_id": False}).limit(10)
        self.data = self.__cursor_to_df(cursor)

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
