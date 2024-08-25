# Base Dependencies
# ----------------
import numpy as np
from typing import List, Any, Optional

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator

# Constants
# ---------
from constants import DATASETS, DDI_IOB_TAGS, N2C2_IOB_TAGS


class IOBFeature(BaseEstimator):
    """
    IOB encoding

    Obtains the IOB tag of each token in the relation's sentence.
    """

    def __init__(self, dataset: str, padding_idx: Optional[int] = None):
        """
        Args:
            dataset (str): dataset name
            padding_idx (int, default = 0): index that will be used for padding
        """
        if dataset not in DATASETS:
            raise ValueError("unsupported dataset '{}'".format(dataset))

        self.dataset = dataset
        self.iob_tags = N2C2_IOB_TAGS if dataset == "n2c2" else DDI_IOB_TAGS
        self.padding_idx = padding_idx

    def get_feature_names(self, input_features=None):
        """
        Gets the name of the feature
        """
        return ["IOB"]

    def create_iob_feature(self, collection: RelationCollection) -> List[List[int]]:
        """
        Computes the IOB encoding for a list of relations.

        Args:
            relations (List[Relation]): list of relations

        Returns:
            IOB encoding of the relations' sentence
        """
        iob_all = []
        o_index = self.iob_index("O")

        for i in range(len(collection)):

            # IOB of entity1
            B_e1 = self.iob_index("B-" + collection.relations[i].entity1.type)
            I_e1 = self.iob_index("I-" + collection.relations[i].entity1.type)
            iob_e1 = [B_e1] + ([I_e1] * (len(collection.entities1_tokens[i]) - 1))

            # IOB of entity2
            B_e2 = self.iob_index("B-" + collection.relations[i].entity2.type)
            I_e2 = self.iob_index("I-" + collection.relations[i].entity2.type)
            iob_e2 = [B_e2] + ([I_e2] * (len(collection.entities2_tokens[i]) - 1))

            iob_sent = (
                ([o_index] * len(collection.left_tokens[i]))
                + iob_e1
                + ([o_index] * len(collection.middle_tokens[i]))
                + iob_e2
                + ([o_index] * len(collection.right_tokens[i]))
            )

            iob_all.append(np.array(iob_sent))

        return iob_all

    def iob_index(self, iob_tag: str):
        """
        Computes the index of the corresponding IOB tag
        """
        idx = self.iob_tags.index(iob_tag)

        if self.padding_idx is not None and idx >= self.padding_idx:
            idx += 1
        return idx

    def fit(self, x: RelationCollection, y: Any = None):
        return self

    def transform(self, x: RelationCollection) -> list:
        return self.create_iob_feature(x)

    def fit_transform(self, x: RelationCollection, y: Any = None) -> list:
        return self.create_iob_feature(x)
