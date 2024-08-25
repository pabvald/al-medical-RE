#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on the BiLSTM model and the different datasets (i.e. n2c2, DDI)
"""

# Base Dependencies
# -----------------
from copy import deepcopy
from pathlib import Path
from os.path import join as pjoin

# Package Dependencies
# --------------------
from .common import final_repetition

# Local Dependencies
# ------------------
from training.config import PLExperimentConfig, BaalExperimentConfig
from training.bilstm import BilstmTrainer
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
    BaalQueryStrategy,
    EXP_RANDOM_SEEDS
)

MODEL_NAME = "bilstm"


def bilstm_passive_learning_n2c2(init_repetition: int = 0, n_repetitions: int = 5, logging: bool = True):

    config = PLExperimentConfig(
        max_epoch=25,
        batch_size=32
    )
  
    for rel_type in N2C2_REL_TYPES:
        # load datasets
        train_dataset = load_from_disk(
            str(Path(pjoin(N2C2_HF_TRAIN_PATH, MODEL_NAME, rel_type)))
        )
        test_dataset = load_from_disk(
            str(Path(pjoin(N2C2_HF_TEST_PATH, MODEL_NAME, rel_type)))
        )

        # create trainer
        trainer = BilstmTrainer("n2c2", train_dataset, test_dataset, rel_type)
            
        for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
            # set random seed
            random_seed: int = EXP_RANDOM_SEEDS[repetition]
            set_seed(random_seed)
            config.seed = random_seed
            # train passive learning
            trainer.train_passive_learning(config=config, logging=logging)


def bilstm_passive_learning_ddi(init_repetition:int = 0, n_repetitions: int = 5, logging: bool = True):
    config = PLExperimentConfig(
        max_epoch=25,
        batch_size=32
    )

    # load datasets
    train_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TRAIN_PATH, MODEL_NAME))))
    test_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TEST_PATH, MODEL_NAME))))

    # create trainer
    trainer = BilstmTrainer("ddi", train_dataset, test_dataset)

    # train passive learing
    for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
        # set random seed
        random_seed: int = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed
        
        trainer.train_passive_learning(config=config, logging=logging)


def bilstm_active_learning_n2c2(init_repetition:int = 0, n_repetitions: int = 5, logging: bool = True):

    config = BaalExperimentConfig(
        max_epoch=15,
        batch_size=32,
        all_bayesian=False
    )
    
    
    for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
        # set random seed
        random_seed: int = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        for rel_type in N2C2_REL_TYPES:

            # load datasets
            train_dataset = load_from_disk(
                str(Path(pjoin(N2C2_HF_TRAIN_PATH, MODEL_NAME, rel_type)))
            )
            test_dataset = load_from_disk(
                str(Path(pjoin(N2C2_HF_TEST_PATH, MODEL_NAME, rel_type)))
            )

            # create trainer
            trainer = BilstmTrainer(
                dataset="n2c2",
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                relation_type=rel_type,
            )

            for query_strategy in BaalQueryStrategy:
                exp_config = deepcopy(config)
                trainer.train_active_learning(query_strategy, config=exp_config, logging=logging)


def bilstm_active_learning_ddi(init_repetition:int = 0, n_repetitions: int = 5, logging: bool = True):
    config = BaalExperimentConfig(max_epoch=15,  batch_size=32, all_bayesian=False)

    # load datasets
    train_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TRAIN_PATH, MODEL_NAME))))
    test_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TEST_PATH, MODEL_NAME))))

    # create trainer
    trainer = BilstmTrainer("ddi", train_dataset, test_dataset)

    for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
        # set random seed
        random_seed: int = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        for query_strategy in BaalQueryStrategy:
            exp_config = deepcopy(config)
            trainer.train_active_learning(query_strategy, config=exp_config, logging=logging)
