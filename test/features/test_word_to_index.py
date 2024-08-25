# Base Dependencies
# -----------------
import pytest

# Local Dependencies
# ------------------
from features.word_to_index import WordToIndex
from models.relation_collection import RelationCollection
from utils import read_list_from_file
from vocabulary import Vocabulary

# Constants
# ---------
from constants import N2C2_VOCAB_PATH


# Fixtures
# --------
@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return Vocabulary(read_list_from_file(N2C2_VOCAB_PATH))


# Tests
# ------
def test_word_to_index_init(vocab):
    w2i = WordToIndex(vocab)
    assert isinstance(w2i, WordToIndex)
    assert isinstance(w2i.vocab, Vocabulary)


def test_word_to_index_get_feature_names():
    w2i = WordToIndex(vocab)
    assert w2i.get_feature_names() == ["word_to_index"]


def test_create_word_to_index_feature(n2c2_small_collection, NLP, vocab):
    w2i = WordToIndex(vocab)
    RelationCollection.set_nlp(NLP)

    e1_idx, e2_idx, sent_idx = w2i.word_to_index(n2c2_small_collection)

    assert len(e1_idx) == len(n2c2_small_collection)
    assert len(e1_idx) == len(e2_idx)
    assert len(e1_idx) == len(sent_idx)
