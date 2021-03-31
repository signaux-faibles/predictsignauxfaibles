# pylint: disable=missing-function-docstring
from jsonschema import ValidationError
import pytest

from predictsignauxfaibles.utils import MongoDBQuery, parse_yml_config, ConfigFileError


def test_empty_pipeline():
    query = MongoDBQuery()
    pipeline = query.to_pipeline()
    assert len(pipeline) == 0, isinstance(pipeline, list)


def test_match_pipeline():
    query = MongoDBQuery()
    query.add_standard_match(
        date_min="1999-12-13",
        date_max="2021-12-10",
        min_effectif=10,
        sirets=["123456789", "123451234"],
    )
    pipeline = query.to_pipeline()
    assert len(pipeline) == 1, isinstance(pipeline, list)
    assert "$match" in pipeline[0].keys()


def test_full_query():
    query = MongoDBQuery()
    query.add_standard_match(
        date_min="1999-12-13",
        date_max="2021-12-10",
        min_effectif=10,
        sirets=["123456789", "123451234"],
    )
    query.add_sort()
    query.add_limit(1_000_000)
    query.add_replace_root()
    query.add_projection(fields=["outcome", "periode", "siret"])
    pipeline = query.to_pipeline()
    assert len(pipeline) == 5, isinstance(pipeline, list)
    for i, key in enumerate(("$match", "$sort", "$limit", "$replaceRoot", "$project")):
        assert key in pipeline[i].keys() and len(pipeline[i].keys()) == 1


def test_filter_on_categoricals():
    import itertools

    query = MongoDBQuery()
    query.add_standard_match(
        date_min="1999-12-13",
        date_max="2021-12-10",
        min_effectif=10,
        sirets=["44937840500012"],
        categorical_filters={"region": ["Bourgogne-Franche-Comté"]},
    )
    pipeline = query.to_pipeline()
    assert "$match" in pipeline[0].keys()
    test_filters = [filter.keys() for filter in pipeline[0]["$match"]["$and"]]
    assert "value.region" in list(itertools.chain(*test_filters))


def test_same_stage_multiple_times():
    query = MongoDBQuery()
    for i in range(1, 4):
        query.add_limit(limit=i)
        assert len(query.to_pipeline()) == i


def test_reset_pipeline():
    query = MongoDBQuery()
    for i in range(1, 4):
        query.add_limit(limit=i)
        assert len(query.to_pipeline()) == 1
        query.reset()


def test_parse_yaml_conf_ok():
    conf = parse_yml_config("./tests/fake_data/correct.yml")
    assert isinstance(conf, dict)


def test_parse_yaml_conf_missing_keys():
    with pytest.raises(ValidationError):
        parse_yml_config("./tests/fake_data/missing_entry.yml")


def test_parse_yaml_conf_file_not_found():
    with pytest.raises(ConfigFileError):
        parse_yml_config("./tests/fake_data/does_not_exist.docx")
