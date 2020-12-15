# TODO: find better name for module ?

from abc import ABC, abstractmethod
import logging
from typing import List

import pandas as pd


class SFModel(ABC):
    """
    Main abstract base class for a Signaux Faibles model.
    Used to define the main interfaces that a model must implement.
    """

    def __init__(self):
        self.data = {"training": None, "testing": None}
        self.model = None
        self.features = []
        self.target = None
        self._is_trained = False
        self._performance = None

    @classmethod
    def from_config_file(cls, path_to_file: str):
        """
        Instantiate a SFModel instance using a standardized config file template (e.g. a .yaml file)
        """
        # parse information needed to instantiate SFModel from config file
        logging.info(f"Instantiating SFModel with config found in {path_to_file}")
        # return the instantiated SFModel object
        return cls()

    @abstractmethod
    def get_data(
        self,
        sirens: List = None,
        sirets: List = None,
    ):
        """
        Populate the data field with data from prod.db.Features
        """

    @abstractmethod
    def train(self):
        """
        Train a model using training data.
        """

    @abstractmethod
    def evaluate(self):
        """
        Evaluate a model's performance using testing data.
        """

    @abstractmethod
    def predict(self, new_data: pd.DataFrame):
        """
        Given new data in a DataFrame and a trained model, make prediction for each new observation.
        """

    def save_model(self):
        """
        Serialize the model to a format that can be saved and version-controlled for easy reloading.
        `pickle` could work, but I think some much better alternatives exist.
        We should brainstorm this.
        """
        logging.info("Saving model.")
        logging.info("Model saved in /here/model.pickle")
        return self
