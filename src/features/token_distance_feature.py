# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class TokenDistanceFeature(BaseEstimator):
    """
    TokenDistanceFeature

    Computes the number of tokens between the two entities of a relation.
    
    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approach
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["token_dist"]

    def create_token_distance_feature(
        self, collection: RelationCollection
    ) -> numpy.array:
        features = []
        # max = 1
        for doc in collection.middle_tokens:
            features.append([len(doc)])
            # if len(r.middle_context) > max:
            #     max = len(r.middle_context)

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_token_distance_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_token_distance_feature(x)
