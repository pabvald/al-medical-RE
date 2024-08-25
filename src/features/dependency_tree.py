# Base Dependencies
# ----------------
import numpy as np

# Local Dependencies
# ------------------
from models import RelationCollection
from nlp_pipeline import get_pipeline

# 3rd-Party Dependencies
# ----------------------
import networkx as nx
from sklearn.base import BaseEstimator



class DependencyTree(BaseEstimator):
    """
    Dependency Tree

    Computes the dependency tree of each relation
    """

    def __init__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["dependency_tree"]

    def create_dependency_tree(
        self,
        collection: RelationCollection,
    ) -> list:
        features = []
        NLP = get_pipeline()
        parser = NLP.get_pipe("parser")
        for doc in parser.pipe(collection.tokens):
            # build dependency tree
            edges = []
            for sent in doc.sents:
                for token in sent:
                    for child in token.children:
                        edges.append(
                            (
                                "{0}-{1}".format(token.i, token.lower_),
                                "{0}-{1}".format(child.i, child.lower_),
                            )
                        )
                        edges.append(
                            (
                                "{0}-{1}".format(child.i, child.lower_),
                                "{0}-{1}".format(token.i, token.lower_),
                            )
                        )

            T = nx.Graph(edges)
            features.append(T)

        return features

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> list:
        return self.create_dependency_tree(x)

    def fit_transform(self, x: RelationCollection, y=None) -> list:
        return self.create_dependency_tree(x)
