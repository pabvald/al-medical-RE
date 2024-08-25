# coding: utf-8

# Base Dependencies
# ------------------
from pathlib import Path
from os.path import join as pjoin


DDI_VOCAB_PATH = Path(pjoin("data", "ddi", "vocab", "vocab.txt"))
DDI_PATH = Path("data/ddi")
DDI_ENTITY_TYPES = ["DRUG", "GROUP", "BRAND", "DRUG_N"]
DDI_NO_REL = "NO-REL"
DDI_REL_TYPES = ["EFFECT", "MECHANISM", "ADVISE", "INT"]
DDI_ALL_TYPES = [DDI_NO_REL] + DDI_REL_TYPES
DDI_REL_TEST_COUNTS ={"EFFECT": 360, "MECHANISM": 302, "ADVISE": 221, "INT": 96}
DDI_REL_TEST_WEIGHTS = [
    360/979,
    302/979,
    221/979,
    96/979,
]

DDI_ATTR_ENTITY_CANDIDATES = {
    "DRUG": DDI_ENTITY_TYPES,
    "GROUP": DDI_ENTITY_TYPES,
    "BRAND": DDI_ENTITY_TYPES,
    "DRUG_N": DDI_ENTITY_TYPES,
}

DDI_IOB_TAGS = [
    "O",
    "B-DRUG",
    "I-DRUG",
    "B-GROUP",
    "I-GROUP",
    "B-BRAND",
    "I-BRAND",
    "B-DRUG_N",
    "I-DRUG_N",
]

DDI_RD_MAX = 20

DDI_HF_TRAIN_PATH = "data/ddi/train.hf"
DDI_HF_TEST_PATH = "data/ddi/test.hf"
