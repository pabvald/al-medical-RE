# Base Dependencies
# ----------------
import pytest
import logging 

# Local Dependencies
# ------------------
from features import EntityEmbedding
from models import RelationCollection 



# Tests 
# ------
def test_entity_embedding_init(wv_model):
    ent_emb = EntityEmbedding("n2c2", wv_model)
    
    assert isinstance(ent_emb, EntityEmbedding)
    assert ent_emb.dataset == "n2c2"
    assert ent_emb.model == wv_model
    

def test_entity_embedding_init_unsupported_dataset_raises(wv_model):
    with pytest.raises(ValueError):
        ent_emb = EntityEmbedding("i2b2", wv_model)
        

def test_entity_embedding_get_feature_names(wv_model):
    ent_emb = EntityEmbedding("n2c2", wv_model)
    assert ent_emb.get_feature_names() == ["entity_embedding"]
    
    
def test_entity_embedding_fit(n2c2_small_collection, wv_model):
    ent_emb = EntityEmbedding("n2c2", wv_model)
    
    ent_emb = ent_emb.fit(n2c2_small_collection)
    
    assert isinstance(ent_emb, EntityEmbedding)
    
    
def test_entity_embedding_create_entity_embedding(n2c2_small_collection, wv_model, NLP):
    RelationCollection.set_nlp(NLP)
    ent_emb = EntityEmbedding("n2c2", wv_model)
    e1_emb, e2_emb = ent_emb.create_entity_embedding(n2c2_small_collection)
    
    assert e1_emb.shape == e2_emb.shape 
    assert e1_emb.shape[0] == len(n2c2_small_collection)
    assert e1_emb.shape[1] == wv_model.vector_size 
    
    