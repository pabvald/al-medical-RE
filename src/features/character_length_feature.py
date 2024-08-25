# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class CharacterLengthFeature(BaseEstimator):
    """
    CharacterLengthFeature

    Computes the number of characters of each relation's sentence. Used for evaluation.
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["char_length"]

    def create_character_length_feature(
        self, collection: RelationCollection
    ) -> numpy.array:
        features = []
        for r in collection.relations:
            features.append([len(r.text)])

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_character_length_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_character_length_feature(x)
