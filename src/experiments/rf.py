#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on the Random Forest model and the different datasets (i.e. n2c2, DDI)
"""

# Package Dependencies
# --------------------
from .common import final_repetition

# Local Dependencies
# ------------------
from models import RelationCollection
from training.base import ALExperimentConfig
from training.rf import RandomForestTrainer
from training.config import PLExperimentConfig, ALExperimentConfig
from utils import set_seed

# Constants
# ----------
from constants import N2C2_REL_TYPES, EXP_RANDOM_SEEDS, RFQueryStrategy


# Experiments
# -----------
def rf_passive_learning_n2c2(init_repetiton: int = 0, n_repetitions: int = 5, logging: bool = True):
    """
    Model: Random Forest
    Dataset: n2c2
    Learning: passive
    """

    collections = RelationCollection.load_collections("n2c2", splits=["train", "test"])
    config = PLExperimentConfig()

    for repetition in range(init_repetiton, final_repetition(init_repetiton, n_repetitions)):
        # set random seed 
        random_seed = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        for rel_type in N2C2_REL_TYPES:
            train_collection = collections["train"].type_subcollection(rel_type)
            test_collection = collections["test"].type_subcollection(rel_type)

            trainer = RandomForestTrainer(
                dataset="n2c2",
                train_dataset=train_collection,
                test_dataset=test_collection,
                relation_type=rel_type,
            )

            trainer.train_passive_learning(config=config, logging=logging)


def rf_passive_learning_ddi(init_repetiton: int = 0, n_repetitions: int = 5, logging: bool = True):
    """
    Model: Random Forest
    Dataset: DDI
    Learning: passive
    """
    collections = RelationCollection.load_collections("ddi", splits=["train", "test"])
    train_collection = collections["train"]
    test_collection = collections["test"]
    config = PLExperimentConfig()

    trainer = RandomForestTrainer(
        dataset="ddi",
        train_dataset=train_collection,
        test_dataset=test_collection,
    )
    
    for repetition in range(init_repetiton, final_repetition(init_repetiton, n_repetitions)):
        # set random seed 
        random_seed = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed
        trainer.train_passive_learning(config=config, logging=logging)


def rf_active_learning_n2c2(init_repetiton: int = 0, n_repetitions: int = 5, logging: bool = True):
    """
    Model: Random Forest
    Dataset: n2c2
    Learning: active
    """
    collections = RelationCollection.load_collections("n2c2", splits=["train", "test"])
    config = ALExperimentConfig()

    for repetition in range(init_repetiton, final_repetition(init_repetiton, n_repetitions)):
        # set random seed 
        random_seed = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        for rel_type in N2C2_REL_TYPES:
            train_collection = collections["train"].type_subcollection(rel_type)
            test_collection = collections["test"].type_subcollection(rel_type)

            trainer = RandomForestTrainer(
                dataset="n2c2",
                train_dataset=train_collection,
                test_dataset=test_collection,
                relation_type=rel_type,
            )
            for query_strategy in RFQueryStrategy:
                trainer.train_active_learning(query_strategy, config, logging=logging)


def rf_active_learning_ddi(init_repetiton: int = 0, n_repetitions: int = 5, logging: bool = True):
    """
    Model: Random Forest
    Dataset: DDI
    Learning: active
    """
    collections = RelationCollection.load_collections("ddi", splits=["train", "test"])
    train_collection = collections["train"]
    test_collection = collections["test"]

    config = ALExperimentConfig()

    for repetition in range(init_repetiton, final_repetition(init_repetiton, n_repetitions)):
        # set random seed 
        random_seed = EXP_RANDOM_SEEDS[repetition]
        set_seed(random_seed)
        config.seed = random_seed

        trainer = RandomForestTrainer(
            dataset="ddi",
            train_dataset=train_collection,
            test_dataset=test_collection,
        )
        for query_strategy in RFQueryStrategy:
            trainer.train_active_learning(query_strategy, config, logging=logging)
