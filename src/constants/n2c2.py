# coding: utf-8

# Base Dependencies
# -----------------
import re
from pathlib import Path
from os.path import join as pjoin


N2C2_VOCAB_PATH = Path(pjoin("data", "n2c2", "vocab", "vocab.txt"))

N2C2_PATH = Path("data/n2c2")
N2C2_ENTITY_TYPES = [
    "Drug",
    "Strength",
    "Duration",
    "Route",
    "Form",
    "ADE",
    "Dosage",
    "Reason",
    "Frequency",
]
N2C2_ATTR_TYPES = [
    "Strength",
    "Duration",
    "Route",
    "Form",
    "ADE",
    "Dosage",
    "Reason",
    "Frequency",
]

N2C2_REL_TYPES = [
    "Strength-Drug",
    "Duration-Drug",
    "Route-Drug",
    "Form-Drug",
    "ADE-Drug",
    "Dosage-Drug",
    "Reason-Drug",
    "Frequency-Drug",
]

N2C2_REL_TEST_WEIGHTS = [
    10255 / 41086,
    568 / 41086,
    6784 / 41086,
    5382 / 41086,
    981 / 41086,
    3563 / 41086,
    4335 / 41086,
    9218 / 41086, 
]

N2C2_ATTR_ENTITY_CANDIDATES = {
    "Strength": ["Drug"],
    "Duration": ["Drug"],
    "Route": ["Drug"],
    "Form": ["Drug"],
    "ADE": ["Drug"],
    "Dosage": ["Drug"],
    "Reason": ["Drug"],
    "Frequency": ["Drug"],
}

N2C2_SPLITS_DIR = "data/n2c2/splits"
N2C2_HF_TRAIN_PATH = "data/n2c2/train.hf"
N2C2_HF_TEST_PATH = "data/n2c2/test.hf"


N2C2_ANNONYM_PATTERNS = {
    "hour": re.compile(r"\[\*\*\d+-\d+\*\*\]\s*PM"),
    "date": re.compile(
        r"(\[\*\*(Date|Month|Year)[^\*]*\*\*\])|(\[\*\*\d+-\d+-?\d*\*\*\])"
    ),
    "hospital": re.compile(r"(\[\*\*[^\*]*Hospital[^\*]*\*\*\])"),
    "name": re.compile(r"(\[\*\*[^\*]*(Name|name)[^\*]*\*\*\])"),
    "telephone": re.compile(r"(\[\*\*[^\*]*(Telephone|telephone)[^\*]*\*\*\])"),
    "location": re.compile(r"(\[\*\*[^\*]*(Location|location|\d+-/\d+)[^\*]*\*\*\])"),
    "address": re.compile(r"(\[\*\*[^\*]*(Address|address|Country|State)[^\*]*\*\*\])"),
    "age": re.compile(r"(\[\*\*[^\*]*(Age)[^\*]*\*\*\])"),
    "number": re.compile(
        r"(\[\*\*[^\*]*(Number|Numeric Identifier|number)[^\*]*\*\*\])|(\[\*\*\d+\*\*\])"
    ),
}

N2C2_IOB_TAGS = [
    "O",
    "B-Drug",
    "I-Drug",
    "B-Strength",
    "I-Strength",
    "B-Duration",
    "I-Duration",
    "B-Route",
    "I-Route",
    "B-Form",
    "I-Form",
    "B-ADE",
    "I-ADE",
    "B-Dosage",
    "I-Dosage",
    "B-Reason",
    "I-Reason",
    "B-Frequency",
    "I-Frequency",
]

N2C2_RD_MAX = 30
