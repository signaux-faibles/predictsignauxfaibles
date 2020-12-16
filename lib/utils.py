from typing import NamedTuple, List

from datetime import datetime
import pytz


class MongoParams(NamedTuple):
    """
    MongoDb parameters used in config
    """

    url: str
    db: str
    collection: str


class MongoDBQuery:
    """
    TODO
    """

    def __init__(self):
        self.pipeline = []
        self.match_stage = None
        self.sort_stage = None
        self.limit_stage = None
        self.replace_root_stage = None
        self.project_stage = None

    def add_standard_match(self, date_inf, date_sup, min_effectif, batch):
        """
        TODO
        """
        self.match_stage = {
            "$match": {
                "$and": [
                    {"_id.batch": batch},
                    {
                        "_id.periode": {
                            "$gte": self.__date_to_iso(date_inf),
                            "$lt": self.__date_to_iso(date_sup),
                        }
                    },
                    {"value.effectif": {"$gte": min_effectif}},
                ]
            }
        }

        self.pipeline.append(self.match_stage)

        return self

    def add_sort(self):
        """
        TODO
        """
        self.sort_stage = {"$sort": {"value.random_order": -1}}
        self.pipeline.append(self.sort_stage)
        return self

    def add_limit(self, limit: int):
        """
        TODO
        """
        self.limit_stage = {"$limit": limit}
        self.pipeline.append(self.limit_stage)
        return self

    def add_replace_root(self):
        """
        TODO
        """
        self.replace_root_stage = {"$replaceRoot": {"newRoot": "$value"}}
        self.pipeline.append(self.replace_root_stage)
        return self

    def add_projection(self, fields: List):
        """
        TODO
        """
        self.project_stage = {"$project": {field: 1 for field in fields}}
        self.pipeline.append(self.project_stage)
        return self

    def to_pipeline(self):
        """
        TODO
        """
        return self.pipeline

    @staticmethod
    def __date_to_iso(date: str):
        """
        TODO
        """
        return pytz.utc.localize(datetime.strptime(date, "%Y-%m-%d"))
