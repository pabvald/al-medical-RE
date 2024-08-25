# Base Dependencies
# ----------------
from typing import List, Tuple

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from spacy.tokens import Doc
from sklearn.base import BaseEstimator


class RelativeDistanceFeature(BaseEstimator):
    """Relative Distance Feature
    
    Relative distance encoding for each of the two entities e1, e2 in a relation.
    It marks the positions of all the words in a target entity as 0.
    Every word to its right is assigned an incrementally higher distance number and
    every word to its left is assigned an incrementally lower number.
    """

    def __ini__(self):
        pass

    def get_feature_names(self, input_features=None):
        return ["relative_distance"]

    def relative_distance(self, collection: RelationCollection) -> Tuple[List, List]:
        """

        Args:
            collection (RelationCollection): collection on whose relations the relative
            distance feature is computed

        Returns:
           relative distance of every token with respect to e1 and e2
        """
        e1_rd_all = []
        e2_rd_all = []
        entities1: List[Doc] = collection.entities1_tokens
        entities2: List[Doc] = collection.entities2_tokens
        left: List[Doc] = collection.left_tokens
        middle: List[Doc] = collection.middle_tokens
        right: List[Doc] = collection.right_tokens

        for i in range(len(collection)):
            # relative distance to e1
            before_e1 = list(map(lambda x: x - len(left[i]), range(len(left[i]))))
            e1 = [0] * len(entities1[i])
            after_e1 = list(
                map(
                    lambda x: x + 1,
                    range(len(middle[i]) + len(entities2[i]) + len(right[i])),
                )
            )
            e1_rd = before_e1 + e1 + after_e1

            # relative distance to e2
            before_e2 = list(range(len(left[i]) + len(entities1[i]) + len(middle[i])))
            before_e2 = list(map(lambda x: x - len(before_e2), before_e2))
            e2 = [0] * len(entities2[i])
            after_e2 = list(map(lambda x: x + 1, range(len(right[i]))))
            e2_rd = before_e2 + e2 + after_e2

            e1_rd_all.append(e1_rd)
            e2_rd_all.append(e2_rd)

        assert len(e1_rd_all) == len(collection)
        assert len(e1_rd_all) == len(e2_rd_all)

        return e1_rd_all, e2_rd_all

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(
        self, x: RelationCollection
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        return self.relative_distance(x)

    def fit_transform(
        self, x: RelationCollection, y=None
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        return self.relative_distance(x)
