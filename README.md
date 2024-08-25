# Active Learning for Relation Extraction Task from Medical Text

Master Thesis <br>
M.Sc. in Data Science and Artificial Intelligence <br>
Saarland University <br>

Author: **Pablo Valdunciel Sánchez**


## Table of Contents

- [Overview](#overview)
- [Publication](#publication)
- [Motivation](#motivation)
- [Methods and Results](#method-and-results)
- [Repository Content](#repository-structure)
- [Reproducibility](#reproducibility)
- [Resources](#resources)
- [About](#about)


## Overview

This repository contains the code and resources for my master's thesis titled "Active Learning for Relation Extraction Task from Medical Text." The thesis explores methodologies to optimize Relation Extraction (RE) from medical texts using active learning techniques.

![Demonstration of entities and their annotated relations in the n2c2 corpus (Henry et al., 2020): Each instance may feature multiple entities, and the annotations indicate the presence or absence of a relation between any two entities.](https://github.com/user-attachments/assets/dc3ae018-26f9-4963-ba0b-7eb9b1818a08)

<figcaption style="text-align:center; padding: 100px">
  Figure: Demonstration of entities and their annotated relations in the n2c2 corpus (Henry et al., 2020): Each instance may feature multiple entities, and the annotations indicate the presence or absence of a relation between any two entities.
</figcaption>

## Publication

The research and findings from this master's thesis formed the basis for a workshop paper published in the Proceedings of the 1st Workshop on Uncertainty-Aware NLP (UncertaiNLP 2024).

---

### 📄 Optimizing Relation Extraction in Medical Texts through Active Learning: A Comparative Analysis of Trade-offs

**Authors:** Siting Liang, Pablo Valdunciel Sánchez, Daniel Sonntag  
**Published:** March 2024  
**Journal/Conference:** Proceedings of the 1st Workshop on Uncertainty-Aware NLP (UncertaiNLP 2024)  
**DOI:** [10.18653/v1/2024.uncertainlp-1.3](https://aclanthology.org/2024.uncertainlp-1.3/)

---

## Motivation 
The implementation of Health Information Technology (HIT) has substantially increased in healthcare systems worldwide, with a primary emphasis on the digitisation of medical records into Electronic Health Records (EHRs). EHRs incorporate a vast amount of useful health-related information, including relationships between biomedical entities such as drug-drug interactions, adverse drug events, and treatment efficacy. Biomedical publications also provide essential information regarding newly discovered protein-protein interactions, drug-drug interactions, and other types of biomedical relationships. However, given the vast amount of information available within biomedical and clinical documents, it is impractical for healthcare professionals to process this information manually. Therefore, automatic techniques, such as Biomedical and Clinical Relation Extraction, are necessary. Machine learning techniques have been developed for use in relation extraction tasks in the biomedical and clinical domains. Nevertheless, the annotation process required for these medical corpora is time-consuming and expensive. Active learning (AL) is a cost-effective method that labels only the most informative instances for model learning. This research aims to investigate the annotation costs associated with AL when used for relation extraction from biomedical and clinical texts using various supervised learning methods. 


## Method and Results 
This work explores the applicability of three distinct supervised learning methods from different ML families for relation extraction from biomedical (SemEval-2013, Task 9.2, also known as DDI corpus) and clinical (n2c2 2018, Track 2) texts within an active learning framework. The four methods under consideration are Random Forest (a traditional ML method), BiLSTM-based method (a deep learning-based ML method), and a Clinical BERT-based method (a language model-based ML method) with two variations of input. The AL framework employs random sampling as a baseline query strategy, along with Least Confidence (LC) and two batch-based strategies: BatchLC for Random Forest and BatchBALD for the other methods. The evaluation considers not only the achieved performance with significantly fewer annotated samples compared to a standard supervised learning approach but also the overall cost of the annotation process. This includes measuring the duration of the AL step times (i.e., the time required to query and retrain the model in each AL step) as well as the annotation rates of tokens and characters. An empirical study is conducted to identify the most suitable method for relation extraction in this context.

The findings indicate that AL can achieve comparable performance to traditional supervised approaches in relation extraction from biomedical and clinical documents while utilising significantly fewer annotated data. This reduction in annotated data leads to a cost reduction in annotating a medical text corpus. The LC strategy outperformed the random baseline when applied with the Random Forest and Clinical BERT methods (One-sided Wilcoxon Signed-rank Test, p-values $< 10^{-9}$), whereas the batch-based strategies yielded poor results. We propose that LM-based methods are advantageous for interactive annotation processes due to their effective generalisation across diverse corpora, requiring minimal or no adjustment of input features.

## Repository Structure:

- `./data/`: Contains the two corpora used in the evaluation, the pre-trained word embeddings and the pre-trained Clinical BERT model.

- `./doc/`: Holds documentation related to the project, such as license and created figures.

- `./results/`: Stores the results of experiments, including .csv files with metrics and generated plots.

- `./src/`: The central location for all source files for the project. 

- `./tests/`: Contains PyTest tests for the feature generation and  the data models.

- `./pyproject.toml`: The project configuration file.

- `./requirements.txt`: Lists the packages required to run the dataset preprocessing and active learning experiments.

A more detailed `README` can be found in each of the subdirectories.


## Reproducibility:

### 1. Python Setup

First, create a virtual environment and activate it.
```bash
python -m venv .venv
source .venv/bin/activate # (Linux) .venv/Scripts/activate (Windows)
```

Second, install the dependencies.

```bash
pip install -r requirements.txt
```

### 2. Data download
Go to [data/ddi](data/ddi), [data/n2c2](data/n2c2), [data/bioword2vec](data/bioword2vec/) and [data/ml_models](data/ml_models/) and follow the instructions in the `README` files.


### 3. Data preprocessing

The following code will do the preprocessing of the desired corpus:

```Python
from preprocessing import * 

corpus = "n2c2" # or "ddi"

# Split documents into sentences (only does something for n2c2)
split_sentences(corpus)

# Generate relation collections
collections = generate_relations(corpus, save_to_disk=True)

# Generate statistics
generate_statistics(corpus, collections) # will be printed to console
```

For more information on how the preprocessing is implemented, please refer to the `README` in the  [preprocessing](src/preprocessing/README.md) module.


###  4. Generation of training datasets

To run the experiments with the BiLSTM and BERT-based methods, it is necessary to generate HF Datasets with the precomputed input feature and representations. This datasets are then stored in the `./data/` folder and can be loaded at runtime. Computing this at runtime would be too slow. 

The following code generates the HF Datasets for the desired corpus: 

```Python
# Local Dependencies
# ------------------
from vocab import Vocabulary
from models import RelationCollection
from re_datasets import BilstmDatasetFactory, BertDatasetFactory

corpus = "n2c2" # or "ddi"

# Load collections 
collection = RelationCollection.load_collections(corpus)

# Generate vocabulary
train_collection = collections["train"]
vocab = Vocabulary.create_vocabulary(corpus, train_collection, save_to_disk=True)

# Generate HF Datasets for BiLSTM and store in disk
BilstmDatasetFactory.create_datasets(corpus, collections, vocab)

# Generate HF Datasets for BERT and store in disk
BertDatasetFactory.create_datasets(corpus, collections, vocab)
```
 

### 5. Running experiments

To run the experiments, it is necessary to have the HF Datasets generated in the previous step.

The `./src/experiments` folder contains the code for running the different experiments. It is enough adjust certain varaibles (e.g. number of repetitions, with/without logging, training configuration) before running the corresponding fuction. For example: 

```Python	
# Local Dependencies
from experiments import *

# run experiments with BiLSTM on n2c2
bilstm_passive_learning_n2c2()
bilstm_active_learning_n2c2()

# or run experiments with Paired Clinical BERT on DDI 
bert_passive_learning_ddi(pairs=True)
bert_active_learning_ddi(pairs=True)
```

For more information on how the experimental setting is implemented, please refer to the `README` in the [experiments](src/experiments/README.md) module.

### 6. Tracking experiments 

To track the experiments we have chosen [neptune.ai](https://docs.neptune.ai/). Neptune offers experiment tracking and model registry for machine learning projects in a very easy way, storing all the data online and making it accessible with a simple web interface. 

To use neptune, you need to create an account and create a new project. Then, you need to create a new API token. Once you have created the project, you have to asign the name of the project and the API token in the [config.py](./src/config.py) file to the `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN` variables, respectively. 

```Python
# config.py
NEPTUNE_PROJECT = "your_username/your_project_name"
NEPTUNE_API_TOKEN = "your_api_token"
```

If you don't want to use neptune, you can set the `logging` parameter to `False` when running an experiment. 

```Python
bilstm_passive_learning_n2c2(logging=False)
```


## Resources 

### Corpora
 - [2018 n2c2 callenge](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
 - [DDI Extraction corpus](https://github.com/isegura/DDICorpus)

### Word Embeddings
 - [BioWord2vec](https://github.com/ncbi-nlp/BioWordVec)

### Pre-trained Models
 - [Clinical BERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT?text=He+was+administered+10+mg+of+%5BMASK%5D.)


### Libraries

- NLP:
    - [spaCy](https://spacy.io/)
    - [negspaCy](https://github.com/jenojp/negspacy)
    - [ScispaCy](https://allenai.github.io/scispacy/)
    - [gensim](https://radimrehurek.com/gensim/)
    - [PyRuSH](https://github.com/jianlins/PyRuSH)

<br>

- Machine Learning: 
    - [sklearn](https://scikit-learn.org/stable/)
    - [imbalanced-learn](https://imbalanced-learn.org/stable/)

<br>

- Deep Learning:
    - [PyTorch](https://pytorch.org/)
    - [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/)
    - [HuggingFace Transformers](https://huggingface.co/transformers/)
    - [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
    - [Joey NMT](https://github.com/joeynmt/joeynmt)
<br>

- Active Learning:
    - [Baal](https://baal.readthedocs.io/en/latest/)
    - [modAL](https://modal-python.readthedocs.io/en/latest/)

<br>

- Visualisations 
    - [seaborn](https://seaborn.pydata.org/)
    - [bertviz](https://github.com/jessevig/bertviz)
    - [dtreeviz](https://github.com/parrt/dtreeviz)
    
<br>

- Experiments metadata store: 
    - [neptune.ai](https://neptune.ai/)

    
## About 

This project was developed as part of **Pablo Valdunciel Sánchez**'s master's thesis in the *Data Science and Artificial Intelligence* master's programme at Saarland University (Germany).  The work was carried out in collaboration with the German Research Centre for Artificial Intelligence (DFKI), supervised by **Prof. Dr. Antonio Krüger**,  CEO and scientific director of the DFKI, and **Prof. Dr. Daniel Sonntag**, director of the Interactive Machine Learning (IML) department at DFKI, and advised by **Siting Liang**, researcher in the IML department.
