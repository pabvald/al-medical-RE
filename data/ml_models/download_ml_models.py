#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Base Dependencies
# -----------------
from typing import Optional

# Local Dependencies
# ------------------
from constants import MODELS, MODELS_CACHE_DIR

# Transformers Dependencies
# -------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Auxiliar Functions
# ------------------
def _download_bert_models(cache_dir: Optional[str] = None):
    """Download BERT models from HuggingFace hub"""
    if cache_dir is None:
        cache_dir = MODELS_CACHE_DIR

    for model, model_path in MODELS["bert"].items():
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, cache_dir=cache_dir
        )


# Main Functions
# --------------
def download_ml_models(cache_dir: Optional[str] = None):
    """Download the necessary ML models"""
    _download_bert_models(cache_dir)


if __name__ == "__main__":
    download_ml_models()
