from abc import ABC, abstractmethod
import logging
from typing import List

from pygam import LogisticGAM
import pandas as pd

from lib.data import SFDataset
from lib.utils import parse_yml_config


class SFModel(ABC):
    """
    Main abstract base class for a Signaux Faibles model.
    Used to define the main interfaces that a model must implement.
    """

    def __init__(self, dataset: SFDataset, features: List, target: str):
        self.dataset = dataset
        self.model = None
        self.features = features
        self.target = target
        self.X = self.dataset.data[self.features]  # pylint: disable=invalid-name
        self.y = self.dataset.data[[self.target]]  # pylint: disable=invalid-name
        self._is_trained = False
        self._performance = None

    @classmethod
    def from_config_file(cls, path: str):
        """
        Instantiate a SFModel object via a yaml config file
        Args:
            path: path to config file (typically in ./models/{version}/model.yml)
        """
        # parse information needed to instantiate SFModel from config file
        conf = parse_yml_config(path)
        logging.info(f"Instantiating SFModel with config found in {path}")

        # create dataset via config file
        dataset = SFDataset.from_config_file(path).fetch_data()
        dataset.prepare_data()

        # return the instantiated SFModel object
        return cls(
            dataset=dataset, features=conf["features"], target=conf["target"]
        )  # pylint: disable=no-value-for-parameter

    @abstractmethod
    def train(self):
        """
        Train a model using training data.
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

    def __repr__(self):
        return f"""
SFModel wrapping {self.model}
Predicting target "{self.target}" with {len(self.features)} features
        """

    def __str__(self):
        return self.__repr__()


class SFModelGAM(SFModel):
    """
    Generalised Additive Model (GAM)
    From package `pygam`
    """

    def __init__(self, dataset: SFDataset, features: List, target: str = "outcome"):
        super().__init__(dataset, features, target)
        self.model = LogisticGAM()

    def train(self):
        """
        Train a GAM model on the data
        """
        self.model.fit(self.X, self.y)
        self._is_trained = True
        return self

    def predict(self, new_data: pd.DataFrame):
        """
        Use the trained model to make predictions on new data
        """
        return self.model.predict(new_data)

    def predict_proba(self, new_data: pd.DataFrame):
        """
        Use the trained model to make predictions on new data
        """
        return self.model.predict_proba(new_data)
