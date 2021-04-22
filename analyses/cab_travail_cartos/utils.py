# pylint: disable="raise-missing-from"
import logging
from datetime import datetime
import pytz
import config as cab_config
from pymongo import MongoClient

import pandas as pd
from config import CODES_REGION

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def str_to_datetime(date_str: str):
    """
    Converts a string in the YYYY-MM-DD format into a datetime
    """
    return pytz.utc.localize(datetime.strptime(date_str, "%Y-%m-%d"))


def datetime_to_str(my_date: datetime):
    """
    Converts a datetime.datetime object into its YYYY-MM-DD string representation
    """
    return my_date.strftime("%Y-%m-%d")


def unravel_features(raveled_df: pd.DataFrame):
    """
    Removes _id and value prefix from Mongo fetch and
    return a single-level pandas DataFrame
    """
    ids = raveled_df["_id"].apply(pd.Series)
    values = raveled_df["value"].apply(pd.Series)
    return pd.concat([ids, values], axis=1)


def load_features_from_mongo(date_min: str, date_max: str, save_to_path: str = None):
    """
    Loads collection Features from MongoDB corresponding to
    date interval [date_min, date_max] (inclusive)
    """
    features_mongo = (
        MongoClient(host="mongodb://labbdd")
        .get_database("prod")
        .get_collection("Features")
    )
    match_stage = {
        "$match": {
            "$and": [
                {
                    "_id.periode": {
                        "$gte": str_to_datetime(date_min),
                        "$lt": str_to_datetime(date_max),
                    },
                    "_id.batch": "2103_0_urssaf",
                },
                # {"value.effectif": {"$gte": min_effectif}},
            ]
        }
    }

    project_stage = {"$project": {field: 1 for field in cab_config.FEATURES_LIST}}
    pipeline = [match_stage, project_stage]

    features_cursor = features_mongo.aggregate(pipeline)
    logging.info("Loading features from MongoDB...")
    features_raveled = pd.DataFrame(features_cursor)
    logging.info("... Success! Unravelling dataframe...")
    features = unravel_features(features_raveled)
    features["siret"] = features.siret.astype(int)
    logging.info("...done!")

    if save_to_path is not None:
        logging.info("Saving fetched Features data to disk")
        features_tosave = features.copy()
        features_tosave.periode = features_tosave.periode.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        try:
            features_tosave.to_json(save_to_path, orient="records", default_handler=str)
            logging.info("Success")
        except:
            raise Exception("Features could not be saved to json")
        finally:
            del features_tosave

    return features


def load_features(
    date_min: str, date_max: str, from_file: bool = True, filepath: str = None
):
    """
    Loads collection Features either from local file or from MongoDB
    """
    if from_file:
        if filepath is None:
            raise Exception(
                "Requesting to load features from file, but no filepath was provided"
            )
        try:
            features = pd.read_json(filepath, orient="records")
            logging.info("Succesfully loaded Features data from %s", filepath)
            features.periode = features.periode.apply(str_to_datetime)
            return features
        except FileNotFoundError:
            logging.warning(
                "Filepath %s was not found on disk. Fetching for MongoDB", filepath
            )

    features = load_features_from_mongo(date_min, date_max, filepath)
    return features


def load_scores_from_mongo(batch_name: str, algo_name: str, save_to_path: bool = True):
    """
    Loads collection Scores from MongoDB corresponding to
    a given batch_name and algo_name
    """
    scores_mongo = (
        MongoClient(host="mongodb://labbdd")
        .get_database("prod")
        .get_collection("Scores")
    )

    match_stage = {
        "$match": {
            "$and": [
                {
                    "batch": batch_name,
                    "algo": algo_name,
                },
            ]
        }
    }
    project_stage = {
        "$project": {
            field: 1
            for field in [
                "siret",
                "batch",
                "algo",
                "periode",
                "alert",
                "score",
                "small_vs_final",
            ]
        }
    }
    pipeline = [match_stage, project_stage]

    scores_cursor = scores_mongo.aggregate(pipeline)
    logging.info("Loading collection Scores from MongoDB...")
    scores = pd.DataFrame(scores_cursor)
    scores["periode"] = scores.periode.dt.tz_localize(None)
    scores.drop(columns=["_id"], inplace=True)

    if save_to_path is not None:
        logging.info("Saving fetched Scores data to disk")
        try:
            scores.to_json(save_to_path, orient="records", default_handler=str)
            logging.info("Success")
        except:
            raise Exception("Fetch of Scores could not be saved to json")

    return scores


def load_scores(
    batch_name: str, algo_name: str, from_file: bool = True, filepath: str = None
):
    """
    Loads collection Scores either from local file or from MongoDB
    """
    if from_file:
        if filepath is None:
            raise Exception(
                "Requesting to load scores from file, but no filepath was provided"
            )
        try:
            scores = pd.read_json(filepath, orient="records")
            logging.info("Succesfully loaded Scores data from %s", filepath)
            scores.periode = scores.periode.apply(str_to_datetime)
            return scores
        except FileNotFoundError:
            logging.warning(
                "Filepath %s was not found on disk. Fetching for MongoDB", filepath
            )

    scores = load_scores_from_mongo(batch_name, algo_name, filepath)
    # scores.periode = scores.periode.apply(str_to_datetime)
    return scores


def map_region_to_code(reg_name):
    """
    Maps a region name to its INSEE code
    """
    reg_code = None
    if reg_name != "" and reg_name is not None:
        reg_code = CODES_REGION[reg_name]
    return reg_code
