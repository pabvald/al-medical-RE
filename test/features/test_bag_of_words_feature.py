# Base Dependencies
# ----------------
import pytest

# Local Dependencies
# ------------------
from features import BagOfWordsFeature



# Tests 
# ------
def test_bag_of_words_init():
    bow = BagOfWordsFeature()
    assert bow.cv is not None
    

def test_bag_of_words_get_feature_names(n2c2_small_collection):
    bow = BagOfWordsFeature()
    bow = bow.fit(n2c2_small_collection)
    assert len(bow.get_feature_names()) == 50


def test_bag_of_words_get_text(n2c2_small_collection):
    bow = BagOfWordsFeature()
    texts = bow.get_text(n2c2_small_collection)

    assert len(texts) == len(n2c2_small_collection)


def test_bag_of_words_fit(n2c2_small_collection):
    bow = BagOfWordsFeature()
    bow  = bow.fit(n2c2_small_collection)
    assert bow is not None 
    assert isinstance(bow, BagOfWordsFeature)


def test_bag_of_words_transform(n2c2_small_collection): 
    bow = BagOfWordsFeature()
    bow  = bow.fit(n2c2_small_collection)
    feature = bow.transform(n2c2_small_collection)
    assert feature.shape[0] == len(n2c2_small_collection)
    assert feature.shape[1] > 0


def test_bag_of_words_fit_transform(n2c2_small_collection): 
    bow = BagOfWordsFeature()
    feature = bow.fit_transform(n2c2_small_collection)
    assert bow is not None
    assert feature.shape[0] == len(n2c2_small_collection)
    assert feature.shape[1] > 0

