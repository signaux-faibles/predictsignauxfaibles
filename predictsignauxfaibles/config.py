import os
from pathlib import Path
from typing import NamedTuple

from dotenv import find_dotenv, load_dotenv

# SETTING ENV VARIABLES
# ======================
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

ENV = os.getenv("ENV", "develop")
MIN_EFFECTIF = int(os.getenv("MIN_EFFECTIF"))
PACKAGE_ROOTDIR = Path(__file__).parent.parent
OUTPUT_FOLDER = os.path.join(PACKAGE_ROOTDIR, "model_runs")
MODEL_FOLDER = os.path.join(PACKAGE_ROOTDIR, "models")

# SETTING DEFAULT MONGODB PARAMS
# ==============================
# MongoDB parameters

MongoParams = NamedTuple(
    "MongoParams", [("url", str), ("db", str), ("collection", str)]
)

MONGODB_PARAMS = MongoParams(
    url=os.getenv("MONGO_URL", "mongodb://localhost"),
    db=os.getenv("MONGO_DB", "prod"),
    collection=os.getenv("MONGO_COLLECTION", "Features"),
)

# SETTING PACKAGE-WIDE DEFAULT VALUES
# ===================================
# Default values for data

DEFAULT_DATA_VALUES = {
    # Outcomes
    "tag_debit": False,
    "tag_default": False,
    # ACOSS
    "montant_part_ouvriere_past_12": 0.0,
    "montant_part_patronale_past_12": 0.0,
    "montant_part_ouvriere_past_6": 0.0,
    "montant_part_patronale_past_6": 0.0,
    "montant_part_ouvriere_past_3": 0.0,
    "montant_part_patronale_past_3": 0.0,
    "montant_part_ouvriere_past_2": 0.0,
    "montant_part_patronale_past_2": 0.0,
    "montant_part_ouvriere_past_1": 0.0,
    "montant_part_patronale_past_1": 0.0,
    "cotisation": 0.0,
    "montant_part_ouvriere": 0.0,
    "montant_part_patronale": 0.0,
    "cotisation_moy12m": 0.0,
    "ratio_dette": 0.0,
    "ratio_dette_moy12m": 0.0,
    # activité partielle
    "apart_heures_autorisees": 0.0,
    "apart_heures_consommees_cumulees": 0.0,
    "apart_heures_consommees": 0.0,
    # effectif
    "effectif": 0,
    "effectif_ent": 0,
}

# Columns for which NA should be ignored
IGNORE_NA = ["time_til_outcome", "code_naf"]
