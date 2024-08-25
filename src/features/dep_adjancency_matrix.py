# Base Dependencies
# ----------------
import numpy as np

# Package Dependencies
# --------------------
from .dependency_tree import DependencyTree

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
import networkx as nx
from sklearn.base import BaseEstimator


class DependencyAdjacencyMatrix(BaseEstimator):
    """
    Dependency Adjancency Matrix

    Computes the adjacency matrix of the dependency tree for each relation
    """

    def __init__(self, selfloops: bool = True, normalization: bool = True):
        self.dep_tree = DependencyTree()
        self.selfloops = selfloops
        self.normalization = normalization

    def get_feature_names(self, input_features=None):
        return ["dependency_adjancency_matrix"]

    def create_dep_adj_matrix(
        self,
        collection: RelationCollection,
    ) -> list:
        features = []
        trees = self.dep_tree.create_dependency_tree(collection)
        for T in trees:
            # compute adjacency matrix
            A = nx.adjacency_matrix(T)

            # add selfloops
            if self.selfloops:
                I = np.identity(n=A.shape[0])
                A = A + I

            # normalize
            if self.normalization:
                A = A / A.sum(axis=0)

            features.append(A)

        return features

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection, y=None) -> list:
        return self.create_dep_adj_matrix(x)

    def fit_transform(self, x: RelationCollection, y=None) -> list:
        return self.create_dep_adj_matrix(x)
