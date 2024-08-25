# Base Dependencies
# ----------------
import pytest
import numpy as np

# Local Dependencies
# ------------------
from features.relative_distance_feature import RelativeDistanceFeature
from models import RelationCollection, Entity, Relation


# Tests
# ------
def test_relative_distance_init():
    rd = RelativeDistanceFeature()
    assert isinstance(rd, RelativeDistanceFeature)


def test_relative_distance_get_feature_names():
    rd = RelativeDistanceFeature()
    assert len(rd.get_feature_names()) == 2


def test_relative_distance_compute_dimensions(n2c2_small_collection, NLP):
    RelationCollection.set_nlp(NLP)
    Relation.set_nlp(NLP)
    Entity.set_nlp(NLP)

    rd = RelativeDistanceFeature()

    e1_distances, e2_distances = rd.relative_distance(n2c2_small_collection)

    assert len(e1_distances) == len(n2c2_small_collection)
    assert len(e1_distances) == len(e2_distances)

    for e1_rd, e2_rd, tokens in zip(
        e1_distances, e2_distances, n2c2_small_collection.tokens
    ):  
        
        assert len(e1_rd) == len(tokens)
        assert len(e2_rd) == len(tokens)


def test_relative_distance_compute_values(NLP, relation):
    RelationCollection.set_nlp(NLP)
    Relation.set_nlp(NLP)
    Entity.set_nlp(NLP)

    rd = RelativeDistanceFeature()
    
    one_collection = RelationCollection(relation)
    e1_distances, e2_distances = rd.relative_distance(one_collection)
    e1_distance = e1_distances[0]
    e2_distance = e2_distances[0]

    assert len(e1_distance) == len(e2_distance)
    assert e1_distance == [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    assert e2_distance == [-6, -5, -4, -3, -2, -1, 0, 0, 1, 2, 3]
