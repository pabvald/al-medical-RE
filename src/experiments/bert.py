#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on the BERT model and the different datasets (i.e. n2c2, DDI)
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

MODEL_NAME = "bert"


def bert_passive_learning_n2c2(init_repetition: int = 0, n_repetitions: int = 5, pairs: bool = False, logging: bool = True):

    config = PLExperimentConfig(
        max_epoch=25, batch_size=32, val_size=0.2, es_patience=3
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
            trainer = BertTrainer(
                dataset="n2c2",
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                pairs=pairs,
                relation_type=rel_type,
            )

            # train passive learning
            trainer.train_passive_learning(config=config, logging=logging)


def bert_active_learning_n2c2(init_repetition: int = 0, n_repetitions: int = 5, pairs: bool = False, logging: bool = True):

    config = BaalExperimentConfig(max_epoch=10, batch_size=32)

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
            trainer = BertTrainer(
                dataset="n2c2",
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                pairs=pairs,
                relation_type=rel_type,
            )

            for query_strategy in BaalQueryStrategy:
                exp_config = deepcopy(config)
                trainer.train_active_learning(query_strategy, exp_config, logging=logging)


def bert_passive_learning_ddi(init_repetition: int = 0, n_repetitions: int = 5, pairs: bool = False, logging: bool = True):

    config = PLExperimentConfig(
        max_epoch=25, batch_size=32, val_size=0.2, es_patience=3
    )

    for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
        # set random seed
        random_seed: int = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        # load datasets
        train_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TRAIN_PATH, MODEL_NAME))))
        test_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TEST_PATH, MODEL_NAME))))

        # create trainer
        trainer = BertTrainer(
            dataset="ddi",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            pairs=pairs,
        )

        # train passive learning
        trainer.train_passive_learning(config=config, logging=logging)


def bert_active_learning_ddi(init_repetition: int = 0, n_repetitions: int = 5, pairs: bool = False, logging: bool = True):

    config = BaalExperimentConfig(max_epoch=15, batch_size=32,)

    for repetition in range(init_repetition, final_repetition(init_repetition, n_repetitions)):
        # set random seed
        random_seed: int = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        # load datasets
        train_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TRAIN_PATH, MODEL_NAME))))
        test_dataset = load_from_disk(str(Path(pjoin(DDI_HF_TEST_PATH, MODEL_NAME))))

        # create trainer
        trainer = BertTrainer(
            dataset="ddi",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            pairs=pairs,
        )

        for query_strategy in BaalQueryStrategy:
            exp_config = deepcopy(config)
            trainer.train_active_learning(query_strategy, exp_config, logging=logging)
