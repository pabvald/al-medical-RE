# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class SentHasButFeature(BaseEstimator):
    """
    SentHasBut Feature

    Determines if a relation contains the word "but".

    Source: 
        Chowdhury and Lavelli (2013) - Exploiting the Scope of Negations and Heterogeneous Features for Relation 
        Extraction: A Case Study for Drug-Drug Interaction Extraction
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["has_but"]

    def compute_sent_has_but(self, collection: RelationCollection) -> numpy.array:
        features = []
        for doc in collection.tokens: 
            feature = 0
            for token in doc: 
                if token.text == "but":
                    feature = 1
                    break          

            features.append([feature])

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.compute_sent_has_but(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.compute_sent_has_but(x)
