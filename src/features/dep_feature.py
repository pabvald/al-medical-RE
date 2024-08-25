# Base Dependencies
# ----------------
from typing import Optional

# Local Dependencies
# ------------------
from models import RelationCollection
from nlp_pipeline import get_pipeline
# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator

# Constants
# ---------
from constants import DEP_TAGS


class DEPFeature(BaseEstimator):
    """
    Dependency Tagging

    Obtains the DEP tags of each token in the relation's sentence
    """

    def __init__(self, padding_idx: Optional[int] = None):
        self.padding_idx = padding_idx

    def get_feature_names(self, input_features=None):
        return ["DEP"]

    def create_dep_feature(self, collection: RelationCollection) -> list:
        all_dep = []

        NLP = get_pipeline()
        parser = NLP.get_pipe("parser")
        for doc in parser.pipe(collection.tokens):
            r_dep = []
            for t in doc:
                r_dep.append(self.dep_index(t.dep_))

            all_dep.append(r_dep)

        return all_dep

    def dep_index(self, dep_tag: str):
        """
        Computes the index of the corresponding POS tag
        """
        idx = DEP_TAGS.index(dep_tag)

        if self.padding_idx is not None and idx >= self.padding_idx:
            idx += 1
        return idx

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> list:
        return self.create_dep_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> list:
        return self.create_dep_feature(x)
