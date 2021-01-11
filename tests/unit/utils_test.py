# pylint: disable=missing-function-docstring
from lib.utils import MongoDBQuery


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
        batch="2012",
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
        batch="2012",
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