# Base Dependencies
# ----------------
import pytest
import numpy as np

# Local Dependencies
# ------------------
from features.pos_feature import POSFeature
from constants import U_POS_TAGS
from models import RelationCollection, Entity
from models.relation import Relation, RelationN2C2


# Tests
# ------
def test_pos_init():
    pos = POSFeature()
    assert isinstance(pos, POSFeature)
    assert pos.padding_idx is None


def test_pos_get_feature_names():
    pos = POSFeature()
    assert pos.get_feature_names() == ["POS"]


def test_pos_fit(n2c2_small_collection):
    pos = POSFeature()
    pos = pos.fit(n2c2_small_collection)
    assert isinstance(pos, POSFeature)


def test_pos_index():
    for padding_idx in [0, 1, 5]:
        pos = POSFeature(padding_idx=padding_idx)

        for tag in U_POS_TAGS:
            idx = pos.pos_index(tag)
            assert idx != padding_idx


def test_pos_index_unknown_tag_raises():
    pos = POSFeature()
    with pytest.raises(Exception):
        idx = pos.pos_index("B-UNKN")


def test_pos_create_pos_feature_dimensions(n2c2_small_collection, NLP):
    RelationCollection.set_nlp(NLP)

    pos = POSFeature()
    pos_feature = pos.create_pos_feature(n2c2_small_collection)
    assert len(pos_feature) == len(n2c2_small_collection)

    for i in range(len(n2c2_small_collection)):
        assert len(pos_feature[i]) == len(n2c2_small_collection.tokens[i])


def test_pos_create_pos_feature_values(relation, NLP):
    RelationCollection.set_nlp(NLP)

    one_collection = RelationCollection(relation)

    pos = POSFeature()
    pos_feature = pos.create_pos_feature(one_collection)
    pos_tags = [
        "PRON",
        "AUX",
        "VERB",
        "PROPN",
        "CCONJ",
        "PROPN",
        "NUM",
        "NOUN",
        "ADP",
        "NUM",
        "NOUN",
    ]
    pos_idx = [pos.pos_index(t) for t in pos_tags]
    assert pos_feature[0] == pos_idx
