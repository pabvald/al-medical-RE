# Base Dependencies
# ----------------
from typing import Optional

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator

# Constants
# ---------
from constants import U_POS_TAGS


class POSFeature(BaseEstimator):
    """
    PoS Tagging

    Obtains the universal POS tag of each token in the relation's sentence.
    """

    def __init__(self, padding_idx: Optional[int] = None):
        self.padding_idx = padding_idx

    def get_feature_names(self, input_features=None):
        return ["POS"]

    def create_pos_feature(self, collection: RelationCollection) -> list:
        all_pos = []

        for doc in collection.tokens:
            r_pos = []
            for t in doc:
                r_pos.append(self.pos_index(t.pos_))

            all_pos.append(r_pos)

        return all_pos

    def pos_index(self, pos_tag: str):
        """
        Computes the index of the corresponding POS tag
        """
        idx = U_POS_TAGS.index(pos_tag)

        if self.padding_idx is not None and idx >= self.padding_idx:
            idx += 1
        return idx

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> list:
        return self.create_pos_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> list:
        return self.create_pos_feature(x)
