# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection
from features.token_distance_feature import TokenDistanceFeature



# Tests
# ------
def test_token_distance_init():
    td = TokenDistanceFeature()
    assert isinstance(td, TokenDistanceFeature)
    
    
def test_token_distance_get_feature_names():
    td = TokenDistanceFeature()
    assert td.get_feature_names() == ["token_dist"]
    
    
def test_create_token_distance_feature(n2c2_small_collection, NLP):
    td = TokenDistanceFeature()
    RelationCollection.set_nlp(NLP)
    
    distances = td.create_token_distance_feature(n2c2_small_collection)
    
    assert distances.shape == (len(n2c2_small_collection), 1)
    for i in range(len(n2c2_small_collection)):
        assert len(n2c2_small_collection.middle_tokens[i]) == distances[i]