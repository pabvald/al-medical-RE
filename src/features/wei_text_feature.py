# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class WeiTextFeature(BaseEstimator):
    """
    WeiTextFeature

    Generates the textual encoding used by [Wei et al. (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7153059/),
    where the entities are replaced by their type within the original text of the sentence.
    
    Source: 
        Wei et al. (2020) - Relation Extraction from Clinical Narratives Using Pre-trained Language Models
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["wei_text"]

    def create_text_feature(self, collection: RelationCollection) -> numpy.array:
        features = []
        for r in collection.relations:
            features.append(
                "{left_context}@{e1_type}${middle_context}@{e2_type}${right_context}".format(
                    left_context=r.left_context,
                    e1_type=r.entity1.type,
                    middle_context=r.middle_context,
                    e2_type=r.entity2.type,
                    right_context=r.right_context,
                )
            )

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> numpy.array:
        return self.create_text_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_text_feature(x)
