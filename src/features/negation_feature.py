# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models.relation_collection import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class NegationFeature(BaseEstimator):
    """
    NegationFeature

    Determines if a relation:
        1. does not contain `no`, `n't` or `not`.
        2. doen't contain  any of the following phrases: "not recommended", "should not be", "must not be"
        3. No target entity mention appears in the sentence after “no”, “n’t” or “not”

    Source:
        Chowdhury and Lavelli (2013) - Exploiting the Scope of Negations and Heterogeneous Features for Relation
        Extraction: A Case Study for Drug-Drug Interaction Extraction
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["no_word", "no_phrase", "no_target"]

    def compute_not_feature(self, collection: RelationCollection) -> numpy.array:
        features = []

        for i in range(len(collection)):
            feature = [1, 1, 1]
            # 1. does not contain “no”, “n’t” or “not”
            for token in collection.tokens[i]:
                if token.lemma_ in ["no", "not"]:
                    feature[0] = 0

            # 2. hasn't any of the following phrases
            for phrase in ["not recommended", "should not be", "must not be"]:
                if phrase in collection.relations[i].text:
                    feature[1] = 0

            # 3. No target entity mention appears in the sentence after “no”, “n’t” or “not”
            for token in collection.left_tokens[i]:
                if token.lemma_ in ["no", "not"]:
                    feature[2] = 0

            features.append(feature)

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.compute_not_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.compute_not_feature(x)
