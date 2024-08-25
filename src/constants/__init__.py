# coding: utf-8

"""
Defining global constants
"""

# Base Dependencies
# -----------------
from enum import Enum
from pathlib import Path

# Constants
# ---------
from .n2c2 import *  # N2C2 Dataset Constants
from .ddi import *  # DDI Dataset Constants


DATASETS_PATHS = {"n2c2": N2C2_PATH, "ddi": DDI_PATH}
DATASETS = list(DATASETS_PATHS.keys())

# experiments' random seeds
EXP_RANDOM_SEEDS = [2, 13, 41, 89, 67]

# vocabulary special tokens
PAD_TOKEN, PAD_ID = "<pad>", 0
BOS_TOKEN, BOS_ID = "<s>", 1
EOS_TOKEN, EOS_ID = "</s>", 2
UNK_TOKEN, UNK_ID = "<unk>", 3

# Word Embeddigns
BIOWORD2VEC_PATH = Path("data/bioword2vec/bio_embedding_extrinsic.txt")

# ML MODELS
CHECKPOINTS_CACHE_DIR =  Path("./cache/checkpoints") 
MODELS_CACHE_DIR =  Path("./cache/models")
MODELS = {"bert": {"clinical-bert": "emilyalsentzer/Bio_ClinicalBERT"}}

# Universal PoS Tagging
# Source: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
U_POS_GLOSSARY = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    "EOL": "end of line",
    "SPACE": "space",
}
U_POS_TAGS = list(U_POS_GLOSSARY.keys())

# Dependency tagging
DEP_GLOSSARY = {
    "ROOT": "root",
    "acl": "clausal modifier of noun (adjectival clause)",
    "acl:relcl": None,
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "amod": "adjectival modifier",
    "amod@nmod": None,
    "appos": "appositional modifier",
    "attr": "attribute",
    "aux": "auxiliary",
    "auxpass": "auxiliary (passive)",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "cc:preconj": None,
    "ccomp": "clausal complement",
    "compound": "compound",
    "compound:prt": None,
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "dative": "dative",
    "dep": "unclassified dependent",
    "det": "determiner",
    "det:predet": None,
    "dobj": "direct object",
    "expl": "expletive",
    "intj": "interjection",
    "mark": "marker",
    "meta": "meta modifier",
    "mwe": None,
    "neg": "negation modifier",
    "nmod": "modifier of nominal",
    "nmod:npmod": None,
    "nmod:poss": None,
    "nmod:tmod": None,
    "nsubj": "nominal subject",
    "nsubjpass": "nominal subject (passive)",
    "nummod": "numeric modifier",
    "parataxis": "parataxis",
    "pcomp": "complement of preposition",
    "pobj": "object of preposition",
    "preconj": "pre-correlative conjunction",
    "predet": None,
    "prep": "prepositional modifier",
    "punct": "punctuation",
    "quantmod": "modifier of quantifier",
    "xcomp": "open clausal complement",
}

DEP_TAGS = list(DEP_GLOSSARY.keys())


# BiLSTM Model
RD_EMB_DIM = 25
IOB_EMB_DIM = 5
BIOWV_EMB_DIM = 200
POS_EMB_DIM = 20
DEP_EMB_DIM = 20

# Active Learning Strategies
class BaalQueryStrategy(Enum):
    RANDOM = "random"
    LC = "least_confidence"
    BATCH_BALD = "batch_bald"

class RFQueryStrategy(Enum):
    RANDOM = "random"
    LC = "least_confidence"
    BATCH_LC = "bach_least_confidence"


# Methods
METHODS_NAMES = {
    "rf": "Random Forest",
    "bilstm": "BiLSTM",
    "bert": "Clinical BERT",
    "bert-pairs": "Paired Clinical BERT",
}
