from predictsignauxfaibles.__main__ import make_parser, ARGS_TO_ATTRS
from predictsignauxfaibles.data import SFDataset, OversampledSFDataset


def test_args_to_attrs_consistency():
    """
    Tests whether optionnal arguments that can be passed to run our package
    are correctly processed to configure our train/test/predict datasets
    """
    args_directory = vars(make_parser().parse_args())
    my_dataset = SFDataset()
    my_os_dataset = OversampledSFDataset(proportion_positive_class=0.3)

    assert (ARGS_TO_ATTRS is not None) and isinstance(ARGS_TO_ATTRS, dict)
    for arg, dataset_attr in ARGS_TO_ATTRS.items():
        assert arg in args_directory.keys()
        if dataset_attr[0] == "train":
            assert hasattr(my_os_dataset, dataset_attr[1])
        else:
            assert hasattr(my_dataset, dataset_attr[1])
