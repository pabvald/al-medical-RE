# Base Dependencies
# ----------------
import pytest

# Local Dependencies
# ------------------
from features import CharDistanceFeature



# Tests 
# ------
def test_bag_of_words_init():
    cd = CharDistanceFeature()
    assert isinstance(cd, CharDistanceFeature)
    

def test_bag_of_words_get_feature_names():
    cd = CharDistanceFeature()
    assert cd.get_feature_names() == ["char_dist"]


def test_create_character_distance_feature(n2c2_small_collection):
    cd = CharDistanceFeature()
    feature = cd.create_character_distance_feature(n2c2_small_collection)

    assert feature.shape[0] == len(n2c2_small_collection)
    assert feature.shape[1] == 1

    for d in feature:
        assert d >= 0

