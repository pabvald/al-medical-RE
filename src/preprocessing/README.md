# Preprocessing

This folder contains the scripts used to generate the candidate entity pairs  and the statistics.

## Pair Generation

In the literature, two main approaches are used to generate the candidate pairs:

1. Sentence splitting + random combination of entities 
2. Heuristic + random combination of entities 

### 1. Sentence splitting + random combination of entities

Sentence splitting is only applied to the **n2c2 corpus**, as the DDI corpus is already split into sentences. 

A clinical document is split into sentences, and all valid combinations of entity pairs within that sentence are considered. This approach has been used by [Wei et al., 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7153059/) and [Christopoulou et al., 2020](https://academic.oup.com/jamia/article/27/1/39/5544735). The following libraries and tools provide sentence splitting of clinical documents: 

- [CLAMP](https://clamp.uth.edu/get-clamp.php) (used by [Wei et al., (2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7153059/))
- [LingPipe](http://www.alias-i.com/lingpipe/) (used by [Christopoulou et al., (2020)](https://academic.oup.com/jamia/article/27/1/39/5544735))
- [PyRuSH Sentecizer](https://github.com/jianlins/PyRuSH) / [Medspacy](https://github.com/medspacy/medspacy)
- [Spark NLP](https://nlp.johnsnowlabs.com/2020/09/13/sentence_detector_dl_healthcare_en.html)
- [ClarityNLP](https://claritynlp.readthedocs.io/en/latest/developer_guide/algorithms/sentence_tokenization.html)
- [Scispacy](https://github.com/allenai/scispacy)

### 2. Heuristic + random combination of entities

A heuristic is used to determine a window in which entities might hold a relation, and all valid combinations of pairs within that window are considered. The following heuristic was first used by [Xu et al., (2018)](https://pubmed.ncbi.nlm.nih.gov/30467557/) and then imitated by [Alimova et al., (2020)](https://www.sciencedirect.com/science/article/pii/S1532046420300095): `the number of characters between the entities is smaller than 1000, and the number of other entities that may participate in relations and locate between the candidate entities is not more than 3`.



### Our Approach

We have chosen the first approach, using [PyRuSH Sentecizer](https://github.com/jianlins/PyRuSH) to split clinical documents into sentences. The main drawback of the second approach is that its heuristic is based on statistics from the whole training set, and considers 1000 characters as the maximum distance between two entities to be considered as candidate pairs. This particular value can result in extreme cases where a relationship occurs between entities in different sentences.

The code for sentence splitting is located in the file `./split_sentences.py` and uses the PyRuSH Sentecizer to split n2c2 `.txt` documents into sentences and store the result in a `.json` file of the same name.

The file `./generate_relations.py` contains the code for preprocessing the corpora and creating a train and test *relation collection*. The train and test relation collections are stored in [datadings](https://datadings.readthedocs.io/en/latest/) for efficient loading. For more information on relation collection implementation, see the [README](../models/README.md).


## Statistics

The file `./generate_statistics.py` contains the code to generate a summary of the corpora, including the number of relations for each type and their proportion to the total.
