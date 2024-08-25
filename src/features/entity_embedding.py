# Base Dependencies
# ----------------
import numpy as np
from typing import List, Tuple

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from gensim.models import KeyedVectors
from spacy.tokens import Doc
from sklearn.base import BaseEstimator

# Constants
# ---------
from constants import DATASETS


class EntityEmbedding(BaseEstimator):
    """
    Entity Embedding

    Obtains the vectors indexes of the two entities in the relation.
    
    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approach
    """

    def __init__(self, dataset: str, model: KeyedVectors):
        if dataset not in DATASETS:
            raise ValueError("unsupported dataset '{}'".format(dataset))
        self.dataset = dataset
        self.model = model

    def get_feature_names(self, input_features=None):
        return ["ent_emb"]

    def create_entity_embedding(
        self, collection: RelationCollection
    ) -> Tuple[np.array, np.array]:
        e1_embs = []
        e2_embs = []
        entities1: List[Doc] = collection.entities1_tokens
        entities2: List[Doc] = collection.entities2_tokens

        assert len(entities1) == len(entities2)

        for e1, e2 in zip(entities1, entities2):
            e1_tokens: List[str] = list(map(lambda t: t.text.lower(), e1))
            e2_tokens: List[str] = list(map(lambda t: t.text.lower(), e2))

            e1_embs.append(self.model.get_mean_vector(e1_tokens))
            e2_embs.append(self.model.get_mean_vector(e2_tokens))

        return np.array(e1_embs), np.array(e2_embs)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> Tuple[np.array, np.array]:
        return self.create_entity_embedding(x)

    def fit_transform(self, x: RelationCollection, y=None) -> Tuple[np.array, np.array]:
        return self.create_entity_embedding(x)
