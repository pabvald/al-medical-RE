# Base Dependencies
# ----------------
import pytest

# Local Dependencies
# ------------------
from models.entity import Entity
from models.relation import Relation, RelationN2C2, RelationDDI


# Fixtures
# --------
@pytest.fixture(scope="session")
def entity1() -> Entity:

    id = "T11"
    text = "Paracetamol"
    type = "Drug"
    doc_id = "doc1202"
    start = 11
    end = start + len(text)
    return Entity(id=id, text=text, type=type, doc_id=doc_id, start=start, end=end)


@pytest.fixture(scope="session")
def entity2() -> Entity:

    id = "T13"
    text = "500mg"
    type = "Dosage"
    doc_id = "doc1202"
    start = 24
    end = start + len(text)
    return Entity(id=id, text=text, type=type, doc_id=doc_id, start=start, end=end)


@pytest.fixture(scope="function")
def relation_attributes(entity1, entity2) -> dict:
    attrs = {
        "doc_id": "doc1202",
        "type": "Dosage-Drug",
        "entity1": entity1,
        "entity2": entity2,
        "label": 1,
        "left_context": "He was administered ",
        "middle_context": " ",
        "right_context": " for three days",
        "middle_entities": [],
    }
    return attrs


@pytest.fixture(scope="function")
def relation(relation_attributes) -> RelationN2C2:
    return Relation(**relation_attributes)


@pytest.fixture(scope="function")
def relation_n2c2(relation_attributes) -> RelationN2C2:
    return RelationN2C2(**relation_attributes)


@pytest.fixture(scope="function")
def relation_unordered(relation_attributes) -> Entity:

    e1 = relation_attributes["entity1"]
    e2 = relation_attributes["entity2"]
    relation_attributes["entity1"] == e2
    relation_attributes["entity2"] == e1

    return Relation(**relation_attributes)


@pytest.fixture(scope="function")
def relation_n2c2_unordered(relation_attributes) -> RelationN2C2:
    e1 = relation_attributes["entity1"]
    e2 = relation_attributes["entity2"]
    relation_attributes["entity1"] == e2
    relation_attributes["entity2"] == e1

    return RelationN2C2(**relation_attributes)


# Tests
# ------
def test_relation_init(entity1, entity2):
    doc_id = "doc1202"
    type = "Dosage-Drug"
    e1 = entity1
    e2 = entity2
    label = 1
    left_context = "He was administered "
    middle_context = " "
    right_context = " for three days"
    middle_entities = []

    relation = Relation(
        doc_id=doc_id,
        type=type,
        entity1=e1,
        entity2=e2,
        label=label,
        left_context=left_context,
        middle_context=middle_context,
        right_context=right_context,
        middle_entities=middle_entities,
    )

    assert isinstance(relation, Relation)
    assert relation.doc_id == doc_id
    assert relation.type == type
    assert relation.entity1 == e1
    assert relation.entity2 == e2
    assert relation.label == label
    assert relation.left_context == left_context
    assert relation.middle_context == middle_context
    assert relation.right_context == right_context
    assert relation.middle_entities == middle_entities


def test_relation_id(relation):
    doc_id = relation.doc_id
    e1_id = relation.entity1.id
    e2_id = relation.entity2.id

    assert relation.id == "{}-{}-{}".format(doc_id, e1_id, e2_id)


def test_relation_text(relation):

    assert relation.text == "He was administered Paracetamol 500mg for three days"


def test_relation_tokens(relation):
    tokens = [t.text for t in relation.tokens]

    assert tokens == [
        "He",
        "was",
        "administered",
        "Paracetamol",
        "500",
        "mg",
        "for",
        "three",
        "days",
    ]


def test_relation_tokens_equal_number(n2c2_collection):
    
    for relation in n2c2_collection.relations:
        assert len(relation.tokens) == (
            len(relation.left_tokens)
            + len(relation.entity1.tokens)
            + len(relation.middle_tokens)
            + len(relation.entity2.tokens)
            + len(relation.right_tokens)
        )   


def test_relation_left_tokens(relation):
    tokens = [t.text for t in relation.left_tokens]
    assert tokens == ["He", "was", "administered"]


def test_relation_middle_tokens(relation):
    tokens = [t.text for t in relation.middle_tokens]
    assert tokens == []


def test_relation_right_tokens(relation):
    tokens = [t.text for t in relation.right_tokens]
    assert tokens == ["for", "three", "days"]


def test_relation_ordered_entities(relation, relation_unordered, entity1, entity2):
    e1, e2 = relation._ordered_entities

    assert e1 == entity1
    assert e2 == entity2

    e1, e2 = relation_unordered._ordered_entities
    assert e1 == entity1
    assert e2 == entity2


def test_relation_subclass_n2c2(relation_n2c2):
    assert issubclass(RelationN2C2, Relation)
    assert isinstance(relation_n2c2, Relation)
    assert isinstance(relation_n2c2, RelationN2C2)


def test_relation_n2c2_ordered_entities(relation_n2c2, relation_n2c2_unordered):

    e1_o, e2_o = relation_n2c2._ordered_entities
    e1_u, e2_u = relation_n2c2_unordered._ordered_entities

    assert isinstance(relation_n2c2, RelationN2C2)
    assert isinstance(relation_n2c2_unordered, RelationN2C2)

    assert e1_o.type != "Drug"
    assert e2_o.type == "Drug"

    assert e1_u.type != "Drug"
    assert e2_u.type == "Drug"
