# Base Dependencies
# ----------------
import pytest
import numpy as np
from random import randrange

# Local Dependencies
# ------------------
from models import RelationCollection
from features.position_feature import PositionFeature

# Constants
# ---------
from constants import (
    N2C2_ATTR_ENTITY_CANDIDATES,
    DDI_ATTR_ENTITY_CANDIDATES,
)


# Tests
# ------
def test_position_init():
    position = PositionFeature("n2c2")
    assert isinstance(position, PositionFeature)
    assert position.dataset == "n2c2"
    assert position.attr_entity_candidates == N2C2_ATTR_ENTITY_CANDIDATES

    position = PositionFeature("ddi")
    assert isinstance(position, PositionFeature)
    assert position.dataset == "ddi"
    assert position.attr_entity_candidates == DDI_ATTR_ENTITY_CANDIDATES


def test_position_init_unsupported_dataset_raises():
    with pytest.raises(ValueError):
        position = PositionFeature("i2b2")


def test_position_get_feature_names():
    position = PositionFeature("n2c2")
    assert len(position.get_feature_names()) == 2


def test_position_fit(n2c2_small_collection):
    position = PositionFeature("n2c2")
    position = position.fit(n2c2_small_collection)
    assert isinstance(position, PositionFeature)


def test_position_create_position_feature_(n2c2_small_collection, relation):

    position = PositionFeature("n2c2")
    position_feature = position.create_position_feature(n2c2_small_collection)
    assert position_feature.shape == (len(n2c2_small_collection), 2)

    for i in range(len(n2c2_small_collection)):
        assert (position_feature[i][0] == 0 and position_feature[i][1] >= 0) or (
            position_feature[i][0] <= 0 and position_feature[i][1] == 0
        )
        
    one_collection = RelationCollection(relation)
    
    position_feature = position.create_position_feature(one_collection)
    
    assert (position_feature == np.array([[-1, 0]])).all()
    
        

