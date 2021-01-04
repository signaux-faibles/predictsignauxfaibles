import os

from dotenv import load_dotenv, find_dotenv

from lib.utils import MongoParams

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

ENV = os.getenv("ENV", "develop")
MONGODB_PARAMS = MongoParams(
    url=os.getenv("MONGO_URL", "mongodb://localhost"),
    db=os.getenv("MONGO_DB", "prod"),
    collection=os.getenv("MONGO_COLLECTION", "Features"),
)

MIN_EFFECTIF = 10
BATCH_ID = "2009_5"
