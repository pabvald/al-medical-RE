# Base Dependencies
# ----------------
import pytest
import numpy as np

# Local Dependencies
# ------------------
from features.dep_feature import DEPFeature
from models import RelationCollection, Entity
from models.relation import Relation, RelationN2C2

# Constants
# ---------
from constants import DEP_TAGS


# Tests
# ------
def test_dep_init():
    dep = DEPFeature()
    assert isinstance(dep, DEPFeature)
    assert dep.padding_idx is None


def test_dep_get_feature_names():
    dep = DEPFeature()
    assert dep.get_feature_names() == ["DEP"]


def test_dep_fit(n2c2_small_collection):
    dep = DEPFeature()
    dep = dep.fit(n2c2_small_collection)
    assert isinstance(dep, DEPFeature)


def test_dep_index():
    for padding_idx in [0, 1, 5]:
        dep = DEPFeature(padding_idx=padding_idx)

        for tag in DEP_TAGS:
            idx = dep.dep_index(tag)
            assert idx != padding_idx


def test_dep_index_unknown_tag_raises():
    dep = DEPFeature()
    with pytest.raises(Exception):
        idx = dep.dep_index("B-UNKN")


def test_dep_create_dep_feature_dimensions(n2c2_small_collection, NLP):
    RelationCollection.set_nlp(NLP)

    dep = DEPFeature()
    dep_feature = dep.create_dep_feature(n2c2_small_collection)
    assert len(dep_feature) == len(n2c2_small_collection)

    for i in range(len(n2c2_small_collection)):
        assert len(dep_feature[i]) == len(n2c2_small_collection.tokens[i])


def test_dep_create_dep_feature_values(relation, NLP):
    RelationCollection.set_nlp(NLP)

    one_collection = RelationCollection(relation)

    dep = DEPFeature()
    dep_feature = dep.create_dep_feature(one_collection)
    dep_tags = [
        "nsubjpass",
        "auxpass",
        "ROOT",
        "compound",
        "cc",
        "conj",
        "nummod",
        "dobj",
        "case",
        "nummod",
        "nmod",
    ]
    dep_idx = [dep.dep_index(t) for t in dep_tags]
    assert dep_feature[0] == dep_idx
