# 3rd-Party Dependencies
# ----------------------
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig

# Constants
# ---------
from constants import MODELS, MODELS_CACHE_DIR


def ClinicalBERT(config: AutoConfig) -> BertForSequenceClassification:
    """Loadas a ClinicalBERT model with the specified number of classes."""
    return BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=MODELS["bert"]["clinical-bert"],
        cache_dir=MODELS_CACHE_DIR,
        config=config,
    )


def ClinicalBERTTokenizer() -> BertTokenizer:
    """Loads a ClinicalBERT tokenizer."""""
    return BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODELS["bert"]["clinical-bert"],
        cache_dir=MODELS_CACHE_DIR,
    )
