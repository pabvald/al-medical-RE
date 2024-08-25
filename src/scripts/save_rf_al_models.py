#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiments on the Random Forest model and the different datasets (i.e. n2c2, DDI)
"""

# Local Dependencies
# ------------------
from models import RelationCollection
from training.base import ALExperimentConfig
from training.random_forest import RandomForestTrainer
from utils import set_seed

# Constants
# ----------
from constants import N2C2_REL_TYPES, EXP_RANDOM_SEEDS, RFQueryStrategy


def main():
    default_config = ALExperimentConfig()
    query_strategy = RFQueryStrategy.LC
    
    # DDI
    collections = RelationCollection.load_collections("ddi", splits=["train", "test"])
    train_collection = collections["train"]
    test_collection = collections["test"]
    

    trainer = RandomForestTrainer(
        dataset="ddi",
        train_dataset=train_collection,
        test_dataset=test_collection,
    )

    set_seed(EXP_RANDOM_SEEDS[0])
    trainer.train_active_learning(
        query_strategy=query_strategy,
        config=default_config,
        save_models=True,
        logging=False,
    )
    
    
    # n2c2 (selected relations)
    collections = RelationCollection.load_collections("n2c2", splits=["train", "test"])
    
    for rel_type in ["Reason-Drug", "ADE-Drug", "Duration-Drug"]:
        train_collection = collections["train"].type_subcollection(rel_type)
        test_collection = collections["test"].type_subcollection(rel_type)

        trainer = RandomForestTrainer(
            dataset="n2c2",
            train_dataset=train_collection,
            test_dataset=test_collection,
            relation_type=rel_type,
        )
        
        set_seed(EXP_RANDOM_SEEDS[0])
        trainer.train_active_learning(
            query_strategy=query_strategy,
            config=default_config,
            save_models=True,
            logging=False,
        )
    
    
if __name__ == "__main__":

   main()
            