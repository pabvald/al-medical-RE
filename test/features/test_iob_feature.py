# Base Dependencies
# ----------------
import pytest
import numpy as np
from random import randrange

# Local Dependencies
# ------------------
from features.iob_feature import IOBFeature
from constants import DDI_IOB_TAGS, N2C2_IOB_TAGS
from models import RelationCollection, Entity, Relation


# Tests
# ------
def test_iob_init():
    iob = IOBFeature("n2c2")
    assert isinstance(iob, IOBFeature)
    assert iob.dataset == "n2c2"
    assert iob.iob_tags == N2C2_IOB_TAGS
    assert iob.padding_idx is None

    iob = IOBFeature("ddi")
    assert isinstance(iob, IOBFeature)
    assert iob.dataset == "ddi"
    assert iob.iob_tags == DDI_IOB_TAGS
    assert iob.padding_idx is None


def test_iob_init_unsupported_dataset_raises():
    with pytest.raises(ValueError):
        iob = IOBFeature("i2b2")


def test_iob_get_feature_names():
    iob = IOBFeature("n2c2")
    assert iob.get_feature_names() == ["IOB"]


def test_iob_fit(n2c2_small_collection):
    iob = IOBFeature("n2c2")
    iob = iob.fit(n2c2_small_collection)
    assert isinstance(iob, IOBFeature)


def test_iob_index():
    for padding_idx in [0, 1, 5]:
        iob = IOBFeature("n2c2", padding_idx=padding_idx)

        for tag in iob.iob_tags:
            idx = iob.iob_index(tag)
            assert idx != padding_idx


def test_iob_index_unknown_tag_raises():
    iob = IOBFeature("n2c2", padding_idx=0)
    with pytest.raises(Exception):
        idx = iob.iob_idx("B-UNKN")


def test_iob_create_iob_feature(n2c2_small_collection, relation):

    iob = IOBFeature("n2c2")
    iob_feature = iob.create_iob_feature(n2c2_small_collection)
    assert len(iob_feature) == len(n2c2_small_collection)

    for i in range(len(n2c2_small_collection)):
        e1_tokens = n2c2_small_collection.entities1_tokens[i]
        e2_tokens = n2c2_small_collection.entities2_tokens[i]
        sent_tokens = n2c2_small_collection.tokens[i]

        assert len(iob_feature[i]) == len(sent_tokens)

        if len(e1_tokens) > 1 and len(e2_tokens) > 1:
            unique = 5
        elif len(e1_tokens) > 1 or len(e2_tokens) > 1:
            unique = 4
        else:
            unique = 3

        assert len(np.unique(iob_feature[i])) == unique

    one_collection = RelationCollection(relation)
    iob_feature = iob.create_iob_feature(one_collection)
    assert len(iob_feature) == 1
    assert list(iob_feature[0]) == [0, 0, 0, 1, 0, 0, 13, 14, 0, 0, 0]
    
