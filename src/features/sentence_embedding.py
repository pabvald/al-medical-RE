# Base Dependencies
# ----------------
import numpy as np
from typing import List

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator


class SentenceEmbedding(BaseEstimator):
    """
    Sentence Embedding

    Obtains the word embedding indexes of the sentence.

    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approachFF
    """

    def __init__(self, model: KeyedVectors):
        self.model = model

    def get_feature_names(self, input_features=None):
        return ["sentence_embedding"]

    def create_sentence_embedding(self, collection: RelationCollection) -> np.array:
        sent_embs = []
        for doc in collection.tokens:
            sent_tokens: List[str] = list(map(lambda t: t.text.lower(), doc))
            sent_embs.append(self.model.get_mean_vector(sent_tokens))

        return np.array(sent_embs)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> np.array:
        return self.create_sentence_embedding(x)

    def fit_transform(self, x: RelationCollection, y=None) -> np.array:
        return self.create_sentence_embedding(x)
