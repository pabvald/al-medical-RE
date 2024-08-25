# Base Dependencies
# ----------------
import numpy as np
from typing import Optional

# Local Dependencies
# ------------------
from models import RelationCollection
from nlp_pipeline import get_pipeline, set_spacy_entities

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator



class NegatedEntitiesFeature(BaseEstimator):
    """
    Negated Entities Feature 
    
    Determines if each of the target entities of a relation is negated or not.
    """

    def __init__(self, padding_idx: Optional[int] = None):
        self.padding_idx = padding_idx

    def get_feature_names(self, input_features=None):
        return ["e1_negated", "e2_negated"]

    def create_negated_entities_feature(self, collection: RelationCollection) -> list:
        features = []

        NLP = get_pipeline()
        parser = NLP.get_pipe("parser")
        negex = NLP.get_pipe("negex")
        docs = collection.tokens 
        
        for i, doc in enumerate(parser.pipe(docs)):
            set_spacy_entities(
                doc,
                collection.left_tokens[i],
                collection.entities1_tokens[i],
                collection.relations[i].entity1.type,
                collection.middle_tokens[i],
                collection.entities2_tokens[i],
                collection.relations[i].entity2.type,
                collection.right_tokens[i],
            )
            assert len(doc.ents) == 2
            doc = negex(doc)
            e1_negated = int(doc.ents[0]._.negex)
            e2_negated = int(doc.ents[1]._.negex)
            
            features.append([e1_negated, e2_negated])

        return np.array(features)

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> list:
        return self.create_negated_entities_feature(x)

    def fit_transform(self, x: RelationCollection, y=None) -> list:
        return self.create_negated_entities_feature(x)
