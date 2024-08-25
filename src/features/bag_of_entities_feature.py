# Base Dependencies
# ----------------
import numpy

# Local Dependencies
# ------------------
from models import RelationCollection
from constants import N2C2_ENTITY_TYPES, DDI_ENTITY_TYPES

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


class BagOfEntitiesFeature(BaseEstimator):
    """
    Bag of Entities

    Computes the count of all entity types between the entities

    Source: 
        Alimova and Tutubalina (2020) - Multiple features for clinical relation extraction: A machine learning approach
    """

    def __init__(self, dataset: str):
        if dataset == "n2c2":
            self.entity_types = N2C2_ENTITY_TYPES
        elif dataset == "ddi":
            self.entity_types = DDI_ENTITY_TYPES
        else:
            raise ValueError(
                "only datasets 'n2c2' and 'ddi' are supported, but no '{}'".format(
                    dataset
                )
            )
        self.dataset = dataset

    def get_feature_names(self, input_features=None):
        names = [] 
        for ent_type in self.entity_types:
            names.append("count_{}".format(ent_type))
        return names

    def create_bag_of_entities_feature(
        self, collection: RelationCollection
    ) -> numpy.array:
        features = []
        for r in collection.relations:
            feature = [0] * len(self.entity_types)
            for e in r.middle_entities:
                if e.type in self.entity_types:
                    feature[self.entity_types.index(e.type)] += 1
            features.append(feature)
        # features /= numpy.max(numpy.abs(features))
        return numpy.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> numpy.array:
        return self.create_bag_of_entities_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        return self.create_bag_of_entities_feature(x)
