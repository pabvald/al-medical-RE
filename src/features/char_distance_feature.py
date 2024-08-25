# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class CharDistanceFeature(BaseEstimator):
    """
    CharDistanceFeature

    Computes the number of characters between the two entities of a relation.
    
    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approach
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["char_dist"]

    def create_character_distance_feature(
        self, collection: RelationCollection
    ) -> numpy.array:
        features = []
        for r in collection.relations:
            features.append([len(r.middle_context)])

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> numpy.array:
        return self.create_character_distance_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_character_distance_feature(x)
