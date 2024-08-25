# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection
from features.wei_text_feature import WeiTextFeature



# Tests
# ------
def test_wei_text_init():
    wei = WeiTextFeature()
    assert isinstance(wei, WeiTextFeature)
    
    
def test_wei_text_get_feature_names():
    wei = WeiTextFeature()
    assert wei.get_feature_names() == ["wei_text"]
    
    
def test_create_wei_text_feature(n2c2_small_collection, relation):
    wei = WeiTextFeature()
    
    texts = wei.create_text_feature(n2c2_small_collection)
    
    assert len(texts) == len(n2c2_small_collection)

    one_collection = RelationCollection(relation)
    text = wei.create_text_feature(one_collection)[0]
    assert str(text) == "He was administered @Drug$ and Paracetamol @Dosage$ for three days"
    