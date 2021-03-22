# pylint: disable=missing-function-docstring
from predictsignauxfaibles.preprocessors import PIPELINE, Preprocessor


def test_final_pipeline():
    for preprocessor in PIPELINE:
        assert isinstance(preprocessor, Preprocessor)
        assert hasattr(preprocessor, "function")
        assert hasattr(preprocessor, "input")
        assert hasattr(preprocessor, "output")
        assert isinstance(preprocessor.input, list)
        assert isinstance(preprocessor.output, list) or preprocessor.output is None
