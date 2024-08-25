# Base Dependencies
# -----------------
from dataclasses import dataclass


@dataclass
class PLExperimentConfig:
    seed: int = 42  # random seed
    max_epoch: int = 25 # maximum num of epochs to train a model 
    batch_size: int = 32 
    val_size: float = 0.2  # size of the validation set
    es_patience: int = 3  # early stopping patience


@dataclass
class ALExperimentConfig(PLExperimentConfig):
    
    batch_size: int = 16
    max_epoch: int = 15  # maximum num of epochs to train on each iteration
    initial_pool_perc: float = 0.025  # initial pool in %
    max_query_size: int = 800
    query_size_perc: float = 0.025  # %
    max_annotation: float = 0.50  # % of the data to be used
    min_train_passes: int = 10


@dataclass
class BaalExperimentConfig(ALExperimentConfig):
    shuffle_prop: float = 0.1
    iterations: int = 5
    max_sample: int = 5000
    all_bayesian: bool = False
