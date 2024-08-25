# Base Dependencies
# ----------------
import numpy as np
from typing import List, Tuple

# Local Dependencies
# ------------------
from vocabulary import Vocabulary
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from spacy.tokens import Doc
from sklearn.base import BaseEstimator


class WordToIndex(BaseEstimator):
    """
    Word to Index

    Obtains the indexes of each word's embedding for entity1, entity2 and the whole
    text of each relation.
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def get_feature_names(self, input_features=None):
        return ["word_to_index"]

    def word_to_index(
        self, collection: RelationCollection
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        e1_tokens = []
        e2_tokens = []
        sent_tokens = []

        entities1: List[Doc] = collection.entities1_tokens
        entities2: List[Doc] = collection.entities2_tokens
        sents: List[Doc] = collection.tokens

        assert len(entities1) == len(entities2)
        assert len(entities1) == len(sents)

        for e1, e2, sent in zip(entities1, entities2, sents):
            e1_tokens.append(list(map(lambda t: t.text.lower(), e1)))
            e2_tokens.append(list(map(lambda t: t.text.lower(), e2)))
            sent_tokens.append(list(map(lambda t: t.text.lower(), sent)))

        return (
            self.vocab.sentences_to_ids(e1_tokens)[0],
            self.vocab.sentences_to_ids(e2_tokens)[0],
            self.vocab.sentences_to_ids(sent_tokens)[0],
        )

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(
        self, x: RelationCollection
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        return self.word_to_index(x)

    def fit_transform(
        self, x: RelationCollection, y=None
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        return self.word_to_index(x)
