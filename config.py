import os

from dotenv import load_dotenv, find_dotenv

from lib.utils import MongoParams

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

ENV = os.getenv("ENV", "develop")

# MongoDB parameters
MONGODB_PARAMS = MongoParams(
    url=os.getenv("MONGO_URL", "mongodb://localhost"),
    db=os.getenv("MONGO_DB", "prod"),
    collection=os.getenv("MONGO_COLLECTION", "Features"),
)

# Other parameters (maybe group them into coherent groups one day...)
MIN_EFFECTIF = os.getenv("MIN_EFFECTIF")
BATCH_ID = os.getenv("BATCH_ID")

# Default values for data
DEFAULT_DATA_VALUES = {
    "montant_part_patronale": 0,
    "montant_part_ouvriere": 0,
    "montant_echeancier": 0,
    "ratio_dette": 0,
    "ratio_dette_moy12m": 0,
    "montant_part_patronale_past_1": 0,
    "montant_part_ouvriere_past_1": 0,
    "montant_part_patronale_past_2": 0,
    "montant_part_ouvriere_past_2": 0,
    "montant_part_patronale_past_3": 0,
    "montant_part_ouvriere_past_3": 0,
    "montant_part_patronale_past_6": 0,
    "montant_part_ouvriere_past_6": 0,
    "montant_part_patronale_past_12": 0,
    "montant_part_ouvriere_past_12": 0,
    "apart_heures_consommees": 0,
    "apart_heures_autorisees": 0,
    "apart_entreprise": 0,
    "tag_default": False,
    "tag_failure": False,
    "tag_outcome": False,
}
