# Base Dependencies
# ----------------
import numpy
from typing import List

# Local Dependencies
# ------------------
from models import RelationCollection

# 3rd-Party Dependencies
# ----------------------
from sklearn.feature_extraction.text import CountVectorizer

# Constants
# ---------
CV_CONFIG = {
    # "min_df": 0.1,
    # "max_df": 0.9,
    "max_features": 50,
}


class BagOfVerbsFeature:
    """
    Bag of Verbs

    All verbs within the middle context of a relation.
    """

    def __init__(self):
        self.cv = CountVectorizer(**CV_CONFIG)

    def get_feature_names(self, input_features=None):
        names = []
        for f in self.cv.get_feature_names():
            names.append("verb_{}".format(f))
        return names

    def get_verbs(self, collection: RelationCollection) -> List[str]:
        contexts = []
        for doc in collection.middle_tokens:
            tokens = []
            for t in doc:
                if t.pos_ == "VERB":
                    tokens.append(t.lemma_)
            tokens = " ".join(tokens)
            contexts.append(tokens)

        return contexts

    def fit(self, x: RelationCollection, y=None):
        texts = self.get_verbs(x)
        self.cv = self.cv.fit(texts)
        return self

    def transform(self, x: RelationCollection, y=None) -> numpy.array:
        texts = self.get_verbs(x)
        X = self.cv.transform(texts)
        X = X.toarray()
        # X = list(X)
        # X /= numpy.max(numpy.abs(X), axis=0)
        return X

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        texts = self.get_verbs(x)
        X = self.cv.fit_transform(texts)
        X = X.toarray()
        # X = list(X)
        # X /= numpy.max(numpy.abs(X), axis=0)
        return X
