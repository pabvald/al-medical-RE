# Base Dependencies
# ----------------
import pytest
from typing import List
from pathlib import Path

# Local Dependencies
# ------------------
from models import Relation, RelationCollection

# 3rd-Party Dependencies
# -----------------------
from spacy.tokens import Doc


# Tests
# ------
def test_relation_collection_init_list(n2c2_relations):

    collection = RelationCollection(n2c2_relations)

    assert isinstance(collection, RelationCollection)
    assert len(collection) == len(n2c2_relations)
    assert collection.relations is not None
    assert collection._tokens is None
    assert collection._left_tokens is None
    assert collection._middle_tokens is None
    assert collection._right_tokens is None
    assert collection._entities1_tokens is None
    assert collection._entities2_tokens is None
    assert RelationCollection.NLP is None


def test_relation_collection_init_single_relation(n2c2_relations):

    collection = RelationCollection(n2c2_relations[0])

    assert isinstance(collection, RelationCollection)
    assert len(collection) == 1


def test_relation_colletion_init_empty_list_raises():

    relations = list()
    with pytest.raises(ValueError):
        collection = RelationCollection(relations)


def test_relation_collection_get_item(n2c2_relations):
    collection = RelationCollection(n2c2_relations)
    collection.compute_all()
    subcollection = collection[[1, 3, 4, 6, 10]]

    assert isinstance(subcollection, RelationCollection)
    assert len(subcollection) == 5
    assert subcollection._tokens is not None
    assert subcollection._left_tokens is not None
    assert subcollection._middle_tokens is not None
    assert subcollection._right_tokens is not None
    assert subcollection._entities1_tokens is not None
    assert subcollection._entities2_tokens is not None


def test_relation_collection_add(n2c2_relations):
    collection1 = RelationCollection(n2c2_relations)
    collection2 = RelationCollection(n2c2_relations)
    collection1.compute_left_tokens()
    collection2.compute_entities2_tokens()

    assert isinstance(collection1, RelationCollection)
    assert isinstance(collection2, RelationCollection)

    collection3 = collection1 + collection2
    assert isinstance(collection3, RelationCollection)
    assert len(collection3) == len(collection1) + len(collection2)
    assert collection3._tokens is None
    assert collection3._left_tokens is not None
    assert collection3._entities1_tokens is None
    assert collection3._middle_tokens is None
    assert collection3._entities2_tokens is not None
    assert collection3._right_tokens is None


def test_relation_collection_ids(n2c2_relations):
    collection = RelationCollection(n2c2_relations)
    ids = collection.ids

    assert len(ids) == len(n2c2_relations)
    for i in range(len(ids)):
        assert ids[i] == collection.relations[i].id


def test_relation_collection_labels(n2c2_relations):
    collection = RelationCollection(n2c2_relations)

    labels = collection.labels

    assert len(labels) == len(n2c2_relations)

    for i in range(len(labels)):
        labels[i] == collection.relations[i].label


def test_relation_collection_tokens(n2c2_relations):
    collection = RelationCollection(n2c2_relations)
    collection.compute_all()
    
    n = len(collection)
    for i in range(n):
        assert len(collection.tokens[i]) == (
            len(collection.left_tokens[i])
            + len(collection.entities1_tokens[i])
            + len(collection.middle_tokens[i])
            + len(collection.entities2_tokens[i])
            + len(collection.right_tokens[i])
        )


def test_relation_collection_to_datading(n2c2_relations, tmpdir):
    filepath = tmpdir.join("relations.msgpack")
    collection = RelationCollection(n2c2_relations)

    collection.to_datading(filepath=filepath)
    assert len(tmpdir.listdir()) == 6


def test_relation_collection_from_datading(n2c2_relations, tmpdir):
    filepath = tmpdir.join("relations.msgpack")
    collection = RelationCollection(n2c2_relations)

    collection.to_datading(filepath=filepath)
    collection2 = RelationCollection.from_datading("n2c2", filepath)

    assert len(collection) == len(collection2)
    assert (collection.ids == collection2.ids).all()


def test_relation_collection_type_subcollection(n2c2_relations):
    collection = RelationCollection(n2c2_relations)
    
    collection.compute_all()
    
    subcollection = collection.type_subcollection("Strength-Drug")
    
    assert isinstance(subcollection, RelationCollection)
    assert len(subcollection) > 0
    assert subcollection._tokens is not None 
    assert subcollection._left_tokens is not None 
    assert subcollection._entities1_tokens is not None 
    assert subcollection._middle_tokens is not None 
    assert subcollection._entities2_tokens is not None 
    assert subcollection._right_tokens is not None