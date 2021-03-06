# pylint: disable=missing-function-docstring

from predictsignauxfaibles.utils import MongoDBQuery, set_if_not_none


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
    query_1 = MongoDBQuery()
    query_1.add_standard_match(
        date_min="1999-12-13",
        date_max="2021-12-10",
        min_effectif=10,
        sirets=["123456789", "123451234"],
        categorical_filters={"region": ["Bourgogne-Franche-Comté", "Île-de-France"]},
    )
    pp_1 = query_1.to_pipeline()
    assert "$match" in pp_1[0].keys()
    test_filters = [
        key for dicts in pp_1[0]["$match"]["$and"] for key, val in dicts.items()
    ]
    assert "value.region" in test_filters

    query_2 = MongoDBQuery()
    query_2.add_standard_match(
        date_min="1999-12-13",
        date_max="2021-12-10",
        min_effectif=10,
        sirets=["123456789", "123451234"],
        categorical_filters={"region": ("Bourgogne-Franche-Comté", "Île-de-France")},
    )
    pp_2 = query_2.to_pipeline()
    assert "$match" in pp_2[0].keys()
    test_filters = [
        key for dicts in pp_2[0]["$match"]["$and"] for key, val in dicts.items()
    ]
    assert "value.region" in test_filters


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


def test_set_if_not_none():
    class TestObject:  # pylint: disable=too-few-public-methods
        """
        Test class
        ...
        Attributes
        ----------
        my_attr_1: str
            should be modified  by test
        my_attr_2: str
            should not be modified by test
        """

        def __init__(self):
            self.my_attr_1 = "foo"
            self.my_attr_2 = 1

    my_obj = TestObject()
    set_if_not_none(my_obj, "my_attr_1", "bar")
    set_if_not_none(my_obj, "my_attr_2", None)

    assert my_obj.my_attr_1 == "bar"
    assert my_obj.my_attr_2 == 1
