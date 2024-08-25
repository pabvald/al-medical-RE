# Base Dependencies
# -----------------
from os.path import join as pjoin
from pathlib import Path

# Local Dependencies
# ------------------
from training.bilstm import BilstmTrainer
from ml_models.bert import ClinicalBERT, ClinicalBERTConfig

# 3rd-Party Dependencies
# ----------------------
from datasets import load_from_disk
from torchinfo import summary

# Constants
# ----------
from constants import (
    N2C2_HF_TRAIN_PATH,
    N2C2_HF_TEST_PATH,
    DDI_HF_TRAIN_PATH,
    DDI_HF_TEST_PATH,
)


def count_parameters():
    """Computes the number of parameters of the BiLSTM model and the Clinical BERT model"""

    # count parameters of BiLSTM model for n2c2 corpus
    print("\n**** BiLSTM - n2c2 Corpus ****")
    rel_type = "Duration-Drug"
    train_dataset = load_from_disk(Path(pjoin(N2C2_HF_TRAIN_PATH, "bilstm", rel_type)))
    test_dataset = load_from_disk(Path(pjoin(N2C2_HF_TEST_PATH, "bilstm", rel_type)))
    trainer = BilstmTrainer("n2c2", train_dataset, test_dataset, rel_type)
    bilstm_model = trainer._init_model()
    summary(bilstm_model)

    # count parameters of BiLSTM model for DDI corpus
    print("\n**** BiLSTM - DDI Corpus ****")
    train_dataset = load_from_disk(Path(pjoin(DDI_HF_TRAIN_PATH, "bilstm")))
    test_dataset = load_from_disk(Path(pjoin(DDI_HF_TEST_PATH, "bilstm")))
    trainer = BilstmTrainer("ddi", train_dataset, test_dataset)
    bilstm_model = trainer._init_model()
    summary(bilstm_model)

    # count parameters of ClinicalBERT model for n2c2 corpus
    print("\n**** ClinicalBERT - n2c2 Corpus ****")
    bert_model = ClinicalBERT(num_classes=2, config=ClinicalBERTConfig)
    summary(bert_model)

    # count parameters of ClinicalBERT model for DDI corpus
    print("\n**** ClinicalBERT - DDI Corpus ****")
    bert_model = ClinicalBERT(num_classes=5, config=ClinicalBERTConfig)
    summary(bert_model)
