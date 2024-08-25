# Base Dependencies
# ----------------
import pytest

# Local Dependencies
# ------------------
from models import Entity


# Tests
# ------
def test_entity_init():
    id = "T11"
    text = "Paracetamol"
    type = "Drug"
    doc_id = "doc1202"
    start = 11
    end = start + len(text)
    entity = Entity(id=id, text=text, type=type,
                    doc_id=doc_id, start=start, end=end)

    assert isinstance(entity, Entity)
    assert entity.id == id
    assert entity.uid == "{}-{}".format(doc_id, id)
    assert entity.type == type
    assert entity.text == text
    assert entity.doc_id == doc_id
    assert entity.start == start
    assert entity.end == end
    assert [t.text for t in entity.tokens] == ["Paracetamol"]


def test_entity_len():
    id = "T11"
    text = "Paracetamol Forte"
    type = "Drug"
    doc_id = "doc1202"
    start = 11
    end = start + len(text)
    entity = Entity(id=id, text=text, type=type,
                    doc_id=doc_id, start=start, end=end)

    assert len(entity) == len(text)


def test_entity_from_n2c2_annotation():

    annotation = "T1	Reason 10179 10197	recurrent seizures"
    doc_id = "doc1020"
    entity = Entity.from_n2c2_annotation(doc_id, annotation)

    assert isinstance(entity, Entity)
    assert entity.id == "T1"
    assert entity.uid == "{}-{}".format(doc_id, "T1")
    assert entity.type == "Reason"
    assert entity.text == "recurrent seizures"
    assert entity.doc_id == doc_id
    assert entity.start == 10179
    assert entity.end == 10197
    assert [t.text for t in entity.tokens] == ["recurrent", "seizures"]


def test_entity_from_ddi_annotation():
    # XML annotation
    #  <entity id="DDI-DrugBank.d519.s3.e0" charOffset="29-36"
    #       type="brand" text="Plenaxis"/>

    # dict annotation obtained from .xml
    annotation = {
        "id": "DDI-DrugBank.d519.s3.e0",
        "charOffset": "29-36",
        "type": "brand",
        "text": "Plenaxis",
    }
    doc_id = "DDI-DrugBank.d519"
    entity = Entity.from_ddi_annotation(doc_id, annotation)

    assert isinstance(entity, Entity)
    assert entity.id == "DDI-DrugBank.d519.s3.e0"
    assert entity.uid == "{}-{}".format(doc_id, "DDI-DrugBank.d519.s3.e0")
    assert entity.type == "BRAND"
    assert entity.text == "Plenaxis"
    assert entity.doc_id == doc_id
    assert entity.start == 29
    assert entity.end == 36 + 1
    assert [t.text for t in entity.tokens] == ["Plenaxis"]


def test_entity_todict():
    id = "T11"
    text = "Paracetamol"
    type = "Drug"
    doc_id = "doc1202"
    start = 11
    end = start + len(text)

    entity = Entity(id=id, text=text, type=type,
                    doc_id=doc_id, start=start, end=end)

    assert entity.todict() == {
        "id": "T11",
        "text": "Paracetamol",
        "type": "Drug",
        "doc_id": "doc1202",
        "start": 11,
        "end": 11 + len("Paracetamol"),
    }
