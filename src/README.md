# Source code 

## Directory structure

- `./constants/`: Contains constants used in the code, such as paths to datasets, glossaries, name mappings, etc.

- `./evaluation/`: Contains code for evaluation, including generation of tables and plots, and computation of statistical tests.

- `./experiments/`: Contains experiments for each method:
  - `./experiments/rf.py`: Random Forest
  - `./experiments/bilstm.py`: BiLSTM
  - `./experiments/bert.py`: ClinicalBERT architectures

- `./extensions/`: Contains extensions to Python Baal and Transformers libraries.

- `./features/`: Contains implementations of input features for RF and BiLSTM methods, and input representations for Clinical BERT and Paired Clinical BERT.

- `./ml_models/`: Implementation of the different machine learning (ML) methods.

- `./models/`: Contains data models for preprocessing and feature generation.

- `./preprocessing/`: Contains preprocessing scripts. 

- `./re_datasets/`: Datasets factory for BiLSTM and BERT models, creating a Hugging Face (HF) Dataset.

- `./scripts/`: Bash scripts for running experiments on the GPU cluster.

- `./training/`: Contains trainers for each ML method and common training resources.

- `./config/`: Configuration of the logging of results to Neptune.ai.

- `./nlp_pipeline.py`: NLP Spacy pipeline.

- `./utils.py`: Helper functions used throughout the code.

- `./vocabulary.py`: Vocabulary module, representing mapping between tokens and indices.


Most of the subdirectories contain a more detailed `README` file.