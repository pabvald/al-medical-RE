# 3rd-Party Dependencies
# ----------------------
from transformers import AutoConfig

# Constants
# ---------
from constants import MODELS, MODELS_CACHE_DIR

ClinicalBERTConfig = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=MODELS["bert"]["clinical-bert"],
    cache_dir=MODELS_CACHE_DIR,
)
ClinicalBERTConfig.hidden_dropout_prob = 0.25
ClinicalBERTConfig.attention_probs_dropout_prob = 0.25
