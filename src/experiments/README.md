# Experiments 

This module contains the code to run the experiments. Each Python file contains the code to run the experiments for a specific method. For example, the `./bilstm.py` file contains the following experiments: 

```Python	
# runs experiments on the n2c2 corpus using the BiLSTM model in a passive learning setting
bilstm_passive_learning_n2c2() 

# runs experiments on the n2c2 corpus using the BiLSTM model in an active learning setting
bilstm_active_learning_n2c2() 

# runs experiments on the DDI corpus using the BiLSTM model in a passive learning setting
bilstm_passive_learning_ddi()

# runs experiments on the DDI corpus using the BiLSTM model in an active learning setting
bilstm_active_learning_ddi()
```

The function `bilstm_active_learning_n2c2()` will run the active learning experiment with the BiLSTM model on the n2c2 corpus by executing the following code: 
    
```Python
# Base Dependencies
# -----------------
from copy import deepcopy
from pathlib import Path
from os.path import join as pjoin

# Local Dependencies
# ------------------
from training.config import BaalExperimentConfig
from training.bilstm import BilstmTrainer
from utils import set_seed

# 3rd-Party Dependencies
# ----------------------
from datasets import load_from_disk

# Constants
# ----------
from constants import (
    N2C2_HF_TRAIN_PATH,
    N2C2_HF_TEST_PATH,
    N2C2_REL_TYPES,
    BAAL_QUERY_STRATEGIES,
    EXP_RANDOM_SEEDS
)
MODEL_NAME = "bilstm"
REPETITIONS = 5
INITIAL_REPETITION = 0
FINAL_REPETITION = INITIAL_REPETITION + REPETITIONS


# Experiment configuration
config = BaalExperimentConfig(
    max_epoch=15,
    batch_size=32
)

# Repetitions of the experiment
for i in range(INITIAL_REPETITION, FINAL_REPETITION):

    # set random seed 
    set_seed(EXP_RANDOM_SEEDS[i])

    # for each relation type
    for rel_type in N2C2_REL_TYPES:

        # load datasets
        train_dataset = load_from_disk(
            Path(pjoin(N2C2_HF_TRAIN_PATH, MODEL_NAME, rel_type))
        )
        test_dataset = load_from_disk(
            Path(pjoin(N2C2_HF_TEST_PATH, MODEL_NAME, rel_type))
        )

        # create trainer
        trainer = BilstmTrainer(
            dataset="n2c2",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            relation_type=rel_type,
        )

        # for each query strategy
        for query_strategy in BAAL_QUERY_STRATEGIES:
            exp_config = deepcopy(config)
            trainer.train_active_learning(query_strategy, exp_config)
```

