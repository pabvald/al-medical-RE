# Base Dependencies
# ----------------
import pytest

# Local Dependencies
# ------------------
from features import BagOfEntitiesFeature

# Constants
# ---------
from constants import DDI_ENTITY_TYPES, N2C2_ENTITY_TYPES

# Tests 
# ------
def test_bag_of_entities_init():
    boe = BagOfEntitiesFeature("n2c2")
    assert boe.dataset == "n2c2"
    assert len(boe.entity_types) > 0
    

def test_bag_of_entities_init_unknown_dataset_raises():
    with pytest.raises(ValueError):
        boe = BagOfEntitiesFeature("i2b2")


def test_bag_of_entities_get_feature_names():
    boe = BagOfEntitiesFeature("ddi")
    assert len(boe.get_feature_names()) ==  len(DDI_ENTITY_TYPES)

    boe = BagOfEntitiesFeature("n2c2")
    assert len(boe.get_feature_names()) ==  len(N2C2_ENTITY_TYPES)
    

def test_bag_of_entities_fit_transform(n2c2_small_collection):
    boe = BagOfEntitiesFeature("n2c2")
    feature  = boe.fit_transform(n2c2_small_collection)

    assert feature.shape[0] == len(n2c2_small_collection)
    assert feature.shape[1] == len(boe.entity_types)

    for i in range(feature.shape[0]):
        feature[i,:].sum() == len(n2c2_small_collection.relations[i].middle_entities)
