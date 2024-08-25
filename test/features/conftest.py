# Base Dependencies 
# -----------------
import pytest
from typing import List
from pathlib import Path

# Local Dependencies
# ------------------
from models.relation import RelationN2C2
from models import Relation, RelationCollection, Entity
from nlp_pipeline import get_pipeline 

# 3rd-Party Dependencies 
# ----------------------
from gensim.models import KeyedVectors


# Fixtures 
# ---------
@pytest.fixture(scope="session")
def n2c2_small_collection() -> RelationCollection:
    small_collection = RelationCollection.from_datading(
        "n2c2", Path("data/n2c2/small/relations.msgpack")
    )
    return small_collection

@pytest.fixture(scope="session")
def wv_model():
    print("Loading bioword2vec model ...")
    model = KeyedVectors.load_word2vec_format("data/bioword2vec/bio_embedding_extrinsic.txt", binary=False)
    print("Bioword2vec loaded!")
    return model 


@pytest.fixture(scope="session")
def NLP():
    return get_pipeline()

# Fixtures
# ---------
@pytest.fixture(scope="session")
def entity1() -> Entity:

    id = "T11"
    text = "Ibuprofen"
    type = "Drug"
    doc_id = "doc1202"
    start = 11
    end = start + len(text)
    return Entity(id=id, text=text, type=type, doc_id=doc_id, start=start, end=end)


@pytest.fixture(scope="session")
def entity2() -> Entity:

    id = "T13"
    text = "Paracetamol"
    type = "Drug"
    doc_id = "doc1202"
    start = 24
    end = start + len(text)
    return Entity(id=id, text=text, type=type, doc_id=doc_id, start=start, end=end)

@pytest.fixture(scope="session")
def entity3() -> Entity:

    id = "T11"
    text = "500mg"
    type = "Dosage"
    doc_id = "doc1202"
    start = 35
    end = start + len(text)
    return Entity(id=id, text=text, type=type, doc_id=doc_id, start=start, end=end)



@pytest.fixture(scope="function")
def relation_attributes(entity1, entity2, entity3) -> dict: 
    attrs = {
        "doc_id": "doc1202",
        "type": "Dosage-Drug",
        "entity1": entity1,
        "entity2": entity3,
        "label": 1,
        "left_context": "He was administered ",
        "middle_context": " and Paracetamol ",
        "right_context": " for three days",
        "middle_entities": [entity2],
    }
    return attrs


@pytest.fixture(scope="function")
def relation(relation_attributes) -> RelationN2C2:
    return RelationN2C2(**relation_attributes)



# @pytest.fixture(scope="session")
# def n2c2_train_collection() -> RelationCollection:

#     collections = RelationCollection.load_collections("n2c2", splits=["train"])
#     collection = collections["train"]
    
#     assert len(collection) > 0
    
#     return collection


# @pytest.fixture(scope="session")
# def ddi_train_collection() -> RelationCollection:

#     collections = RelationCollection.load_collections("ddi", splits=["train"])
#     collection = collections["train"].type_subcollection("Strength-Drug")

#     return collection