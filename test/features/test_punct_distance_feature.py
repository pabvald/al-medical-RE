# Local Dependencies
# ------------------
from features.punct_distance_feature import PunctuationFeature



# Tests
# ------
def test_get_feature_names():
    punct = PunctuationFeature()
    assert punct.get_feature_names() == ["punct_dist"]
    

def test_create_punctuation_distance(n2c2_small_collection):
    punct = PunctuationFeature()
    
    distances = punct.create_punctuation_distance_feature(n2c2_small_collection)
    
    assert distances.shape == (len(n2c2_small_collection), 1)