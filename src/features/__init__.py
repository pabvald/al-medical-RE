# Base Dependencies
# -----------------
import numpy
from typing import Dict

# Package Dependencies
# -----------------
# distance features
from .token_distance_feature import TokenDistanceFeature
from .char_distance_feature import CharDistanceFeature
from .punct_distance_feature import PunctuationFeature
from .position_feature import PositionFeature
from .relative_distance_feature import RelativeDistanceFeature

# word-based features
from .bag_of_entities_feature import BagOfEntitiesFeature
from .bag_of_words_feature import BagOfWordsFeature
from .bag_of_verbs_feature import BagOfVerbsFeature

# text representations
from .wei_text_feature import WeiTextFeature

# embeddings
from .iob_feature import IOBFeature
from .word_to_index import WordToIndex
from .entity_embedding import EntityEmbedding
from .sentence_embedding import SentenceEmbedding

# semantic
from .pos_feature import POSFeature
from .dep_feature import DEPFeature
from .negation_feature import NegationFeature
from .negated_entities_feature import NegatedEntitiesFeature
from .dependency_tree import DependencyTree
from .dep_adjancency_matrix import DependencyAdjacencyMatrix
from .sent_has_but_feature import SentHasButFeature

# others
from .character_length_feature import CharacterLengthFeature
from .token_length_feature import TokenLengthFeature

# Local Dependencies
# -------------------
from models.relation_collection import RelationCollection
from vocabulary import Vocabulary

# 3rd-Party Dependencies
# ----------------------
from sklearn.base import BaseEstimator


# RandomForestFeatures
# --------------------
class RandomForestFeatures(BaseEstimator):
    """Random Forest Features

    Generates the features for the Random Forest model. This features are a subset
    of those used in `Alimova and Tutubalina (2020) - Multiple features for clinical
    relation extraction: A machine learning approach`

    """

    def __init__(self, dataset: str):

        # distance
        self.token_distance_feature = TokenDistanceFeature()
        self.char_distance_feature = CharDistanceFeature()
        self.punctuation_feature = PunctuationFeature()
        self.position_feature = PositionFeature(dataset=dataset)

        # word-base
        self.bag_of_entities_feature = BagOfEntitiesFeature(dataset=dataset)
        self.bag_of_words_feature = BagOfWordsFeature()

    def get_feature_names(self, input_features=None):
        names = []
        names = names + self.token_distance_feature.get_feature_names()
        names = names + self.char_distance_feature.get_feature_names()
        names = names + self.punctuation_feature.get_feature_names()
        names = names + self.position_feature.get_feature_names()
        names = names + self.bag_of_entities_feature.get_feature_names()
        names = names + self.bag_of_words_feature.get_feature_names()
        return names 
    
    def fit(self, x: RelationCollection, y=None):
        # distance
        self.token_distance_feature = self.token_distance_feature.fit(x)
        self.char_distance_feature = self.char_distance_feature.fit(x)
        self.punctuation_feature = self.punctuation_feature.fit(x)
        self.position_feature = self.position_feature.fit(x)

        # word-base
        self.bag_of_entities_feature = self.bag_of_entities_feature.fit(x)
        self.bag_of_words_feature = self.bag_of_words_feature.fit(x)

        return self

    def transform(self, x: RelationCollection) -> numpy.array:
        # distance
        token_distance_feature = self.token_distance_feature.transform(x)
        char_distance_feature = self.char_distance_feature.transform(x)
        punctuation_feature = self.punctuation_feature.transform(x)
        position_feature = self.position_feature.transform(x)

        # word-base
        bag_of_entities_feature = self.bag_of_entities_feature.transform(x)
        bag_of_words_feature = self.bag_of_words_feature.transform(x)

        features = numpy.concatenate(
            (
                token_distance_feature,
                char_distance_feature,
                punctuation_feature,
                position_feature,
                bag_of_entities_feature,
                bag_of_words_feature,
            ),
            axis=1,
        )

        assert features.shape[0] == len(x)

        return features

    def fit_transform(self, x: RelationCollection, y=None) -> numpy.array:
        # distance
        token_distance_feature = self.token_distance_feature.fit_transform(x)
        char_distance_feature = self.char_distance_feature.fit_transform(x)
        punctuation_feature = self.punctuation_feature.fit_transform(x)
        position_feature = self.position_feature.fit_transform(x)

        # word-base
        bag_of_entities_feature = self.bag_of_entities_feature.fit_transform(x)
        bag_of_words_feature = self.bag_of_words_feature.fit_transform(x)

        features = numpy.concatenate(
            (
                token_distance_feature,
                char_distance_feature,
                punctuation_feature,
                position_feature,
                bag_of_entities_feature,
                bag_of_words_feature,
            ),
            axis=1,
        )

        assert features.shape[0] == len(x)

        return features


class RandomForestFeaturesNegation(RandomForestFeatures):
    """Random Forest Features with Negation"""

    def __init__(self, dataset: str):
        super().__init__(dataset)

        # negation
        # self.negation_feature = NegationFeature()
        self.negated_entities = NegatedEntitiesFeature()
        self.has_but = SentHasButFeature()

    def get_feature_names(self, input_features=None):
        names = super().get_feature_names()
        # names = names + self.negation_feature.get_feature_names()
        names = names + self.negated_entities.get_feature_names()
        names = names + self.has_but.get_feature_names()
        return names
    
    def fit(self, x: RelationCollection, y=None):
        super().fit(x)

        # negation
        # self.negation_feature = self.negation_feature.fit(x)
        self.negated_entities = self.negated_entities.fit(x)
        self.has_but = self.has_but.fit(x)

        return self

    def transform(self, x: RelationCollection):
        features = super().transform(x)

        # negation
        # negation_feature = self.negation_feature.transform(x)
        negated_entities = self.negated_entities.transform(x)
        has_but = self.has_but.transform(x)

        features = numpy.concatenate(
            (features, negated_entities, has_but),  # negation_feature,
            axis=1,
        )

        return features

    def fit_transform(self, x: RelationCollection):
        features = super().fit_transform(x)

        # negation
        # negation_feature = self.negation_feature.fit_transform(x)
        negated_entities = self.negated_entities.fit_transform(x)
        has_but = self.has_but.fit_transform(x)

        features = numpy.concatenate(
            (features, negated_entities, has_but),  # negation_feature,
            axis=1,
        )

        return features


# BilstmFeatures
# --------------
class BilstmFeatures(BaseEstimator):
    """BiLSTM Features

    Generates the feautes for the BiLSTM model. These features correspond to
    the ones used in `Hasan et al. - Integrating Text Embedding with Traditional NLP
    Features for Clinical Relation Extraction`
    """

    def __init__(self, dataset: str, vocab: Vocabulary):
        self.dataset = dataset
        self.vocab = vocab

        self.relative_distance = RelativeDistanceFeature()
        self.iob = IOBFeature(dataset, vocab.pad_index)
        self.pos = POSFeature(vocab.pad_index)
        self.dep = DEPFeature(vocab.pad_index)

        self.word2index = WordToIndex(vocab)
        self.char_length = CharacterLengthFeature()

    def fit(self, x: RelationCollection, y=None):

        self.relative_distance = self.relative_distance.fit(x)
        self.iob = self.iob.fit(x)
        self.pos = self.pos.fit(x)
        self.dep = self.dep.fit(x)
        self.word2index = self.word2index.fit(x)

        return self

    def transform(self, x: RelationCollection) -> Dict:
        rd1, rd2 = self.relative_distance.transform(x)
        iob = self.iob.transform(x)
        pos = self.pos.transform(x)
        dep = self.dep.transform(x)
        e1, e2, sent = self.word2index.transform(x)
        seq_length = [len(s) for s in sent]
        char_length = self.char_length.transform(x)

        return {
            "rd1": rd1,
            "rd2": rd2,
            "iob": iob,
            "pos": pos,
            "dep": dep,
            "e1": e1,
            "e2": e2,
            "sent": sent,
            "seq_length": seq_length,
            "char_length": char_length,
        }

    def fit_transform(self, x: RelationCollection, y=None) -> Dict:
        rd1, rd2 = self.relative_distance.fit_transform(x)
        iob = self.iob.fit_transform(x)
        pos = self.pos.fit_transform(x)
        dep = self.dep.fit_transform(x)
        e1, e2, sent = self.word2index.fit_transform(x)
        seq_length = numpy.array([len(s) for s in sent])
        char_length = self.char_length.fit_transform(x)

        return {
            "rd1": rd1,
            "rd2": rd2,
            "iob": iob,
            "pos": pos,
            "dep": dep,
            "e1": e1,
            "e2": e2,
            "sent": sent,
            "seq_length": seq_length,
            "char_length": char_length,
        }


# BertFeatures
# --------------
class BertFeatures(BaseEstimator):
    """BERT Features

    Generates the features for the Bert model.
    """

    def __init__(self):
        self.char_length = CharacterLengthFeature()
        self.token_length = TokenLengthFeature()
        self.wei_text = WeiTextFeature()

    def fit(self, x: RelationCollection, y=None):
        return self

    def transform(self, x: RelationCollection) -> Dict:
        return {
            "sentence": self.wei_text.transform(x),
            "text": [r.text for r in x.relations],
            "char_length": self.char_length.transform(x),
            "seq_length": self.token_length.transform(x),
        }

    def fit_transform(self, x: RelationCollection, y=None) -> Dict:
        return {
            "sentence": self.wei_text.fit_transform(x),
            "text": [r.text for r in x.relations],
            "char_length": self.char_length.fit_transform(x),
            "seq_length": self.token_length.fit_transform(x),
        }
