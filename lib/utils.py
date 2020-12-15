from typing import NamedTuple


class MongoParams(NamedTuple):
    """
    MongoDb parameters used in config
    """

    url: str
    db: str
    collection: str
