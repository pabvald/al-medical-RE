# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class TokenLengthFeature(BaseEstimator):
    """
    TokenLengthFeature

    Computes the number of tokens of each relation. This is used in evaluation.
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["token_length"]

    def create_token_length_feature(
        self, collection: RelationCollection
    ) -> numpy.array:
        features = []
        for doc in collection.tokens:
            features.append([len(doc)])

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_token_length_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_token_length_feature(x)
