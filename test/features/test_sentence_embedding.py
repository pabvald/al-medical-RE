# Local Dependencies
# ------------------
from features.sentence_embedding import SentenceEmbedding



# Tests
# ------
def test_sentence_embedding_init(wv_model):
    se = SentenceEmbedding(wv_model)
    assert isinstance(se, SentenceEmbedding)
    
def test_create_sentence_embedding(n2c2_small_collection, wv_model):
    se = SentenceEmbedding(wv_model)
    
    embeddings = se.create_sentence_embedding(n2c2_small_collection)
    
    assert embeddings.shape == (len(n2c2_small_collection), wv_model.vector_size)