#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on the Random Forest model and the different datasets (i.e. n2c2, DDI)
"""

# Base Dependencies
# -----------------
from copy import deepcopy
from pathlib import Path
from os.path import join as pjoin

# Local Dependencies
# ------------------
from training.config import BaalExperimentConfig
from training.bert import BertTrainer
from utils import set_seed

# 3rd-Party Dependencies
# ----------------------
from datasets import load_from_disk

# Constants
# ----------
from constants import (
    DDI_HF_TEST_PATH,
    DDI_HF_TRAIN_PATH,
    N2C2_HF_TRAIN_PATH,
    N2C2_HF_TEST_PATH,
    N2C2_REL_TYPES,
    EXP_RANDOM_SEEDS,
    BaalQueryStrategy
)


def main():
    
    config = BaalExperimentConfig(max_epoch=15, batch_size=32,)
    query_strategy = BaalQueryStrategy.LC 
    
    # DDI  
    # set random seed
    set_seed(EXP_RANDOM_SEEDS[0])

    # load datasets
    train_dataset = load_from_disk(Path(pjoin(DDI_HF_TRAIN_PATH, "bert")))
    test_dataset = load_from_disk(Path(pjoin(DDI_HF_TEST_PATH, "bert")))

    # create trainer
    trainer = BertTrainer(
        dataset="ddi",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        pairs=False,
    )

    exp_config = deepcopy(config)
    trainer.train_active_learning(
        query_strategy=query_strategy, 
        config=exp_config,
        verbose=True, 
        save_models=True,  
        logging=False
    )
            
    # n2c2 
    set_seed(EXP_RANDOM_SEEDS[0])

    for rel_type in ["Reason-Drug", "Duration-Drug", "ADE-Drug"]:

        # load datasets
        train_dataset = load_from_disk(
            Path(pjoin(N2C2_HF_TRAIN_PATH, "bert", rel_type))
        )
        test_dataset = load_from_disk(
            Path(pjoin(N2C2_HF_TEST_PATH, "bert", rel_type))
        )

        # create trainer
        trainer = BertTrainer(
            dataset="n2c2",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            pairs=False,
            relation_type=rel_type,
        )

        exp_config = deepcopy(config)
        trainer.train_active_learning(
            query_strategy, 
            exp_config, 
            save_models=True,
            logging=False
        )
        