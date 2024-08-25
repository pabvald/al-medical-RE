# Base Dependencies 
# -----------------
import numpy as np
import pytest
from pathlib import Path

# Local Dependencies
# ------------------
from models import RelationCollection


@pytest.fixture(scope="session")
def n2c2_collection() -> RelationCollection:
    collections = RelationCollection.load_collections("n2c2", splits=["train"])
    collection = collections["train"]
    collection = collection[np.random.randint(0, len(collection), 80)]
    return collection

@pytest.fixture(scope="session")
def n2c2_relations(n2c2_collection):
    return n2c2_collection.relations

