"""
Models that are not implemented in scikit-learn must be added to this module, they must
implement the sklearn API for predictors.

See: https://scikit-learn.org/stable/developers/develop.html
"""

from sklearn.base import BaseEstimator


class SFModel(BaseEstimator):
    """Base sklearn estimator."""

    def __init__(self):
        pass

    def fit(self, X, y):  # pylint: disable=invalid-name
        """
        see: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
        """
        raise NotImplementedError()

    def predict(self, X):  # pylint: disable=invalid-name
        """
        see: https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator
        """
        raise NotImplementedError()
