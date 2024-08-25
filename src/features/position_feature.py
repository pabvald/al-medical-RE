# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator

# Constants
# ---------
from constants import (
    N2C2_ATTR_ENTITY_CANDIDATES,
    DDI_ATTR_ENTITY_CANDIDATES,
)


class PositionFeature(BaseEstimator):
    """
    Position Distance

    Computes the position of the entity candidate (drug) with respect to
    the attribute among the entire entity candidates of the attribute, where
    the position of medical attribute is set to 0.
    
    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approach
    """

    def __init__(self, dataset: str):
        if dataset == "n2c2":
            self.attr_entity_candidates = N2C2_ATTR_ENTITY_CANDIDATES

        elif dataset == "ddi":
            self.attr_entity_candidates = DDI_ATTR_ENTITY_CANDIDATES
        else:
            raise ValueError(
                "only datasets 'n2c2' and 'ddi' are supported, but no '{}'".format(
                    dataset
                )
            )
        self.dataset = dataset

    def get_feature_names(self, input_features=None):
        return  ["position_1", "position_2"]

    def create_position_feature(self, collection: RelationCollection) -> numpy.array:
        features = []
        for r in collection.relations:
            feature = [0] * 2

            attr, drug = r._ordered_entities
            candidates = self.attr_entity_candidates[attr.type]

            # count middle entities which could form the same type of relation
            # i.e., count number of middle entities that are drugs for n2c2 and DDI
            position = 0
            for ent in r.middle_entities:
                if ent.type in candidates:
                    position += 1

            ent1 = r.entity1
            ent2 = r.entity2
            # if the attribute is the first entity, the position is positive
            if ent1.type == attr.type:
                feature[0] = 0
                feature[1] = position
            # if the attribute is the second entity, the position is negative
            elif ent2.type == attr.type:
                feature[0] = -position
                feature[1] = 0
            else:
                raise ValueError("none of the entities correspond with the attribute")

            features.append(feature)

        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> numpy.array:
        return self.create_position_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_position_feature(x)
