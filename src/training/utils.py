# Base Dependencies
# -----------------
import numpy as np
from typing import Tuple, Optional, Dict

# 3rd-Party Dependencies
# --------------------
from baal.active.heuristics.heuristics import (
    AbstractHeuristic,
    Random,
    Certainty,
    BatchBALD,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, EvalPrediction


# Query Strategies
# ----------------
def random_sampling(classifier: BaseEstimator, X_pool: list, n_instances: int = 1):
    """Random sampling query strategy for modAL

    Args:
        classifier (BasEstimator): the classifier.
        X_pool (list): the pool of unlabeled instances.
        n_instances (int): number of instances to query. Default to 1.

    Returns:
        list: query indexes, list: query instances
    """
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), n_instances)
    return query_idx, X_pool[query_idx]


def get_baal_query_strategy(
    name: str,
    shuffle_prop: float = 0.0,
    query_size: int = 3,
    reduction: str = "none",
    seed: Optional[int] = None,
) -> AbstractHeuristic:
    """Obtains the desired query strategy (or heuristic) with the provided configuration

    Args:
        name (str): name of the query strategy.
        shuffle_prop (float): shuffle proportion. Default to 0.0.
        query_size (int): number of queries done by BALD and or BatchBALD
        reduction (Union[str, Callable]): Reduction used after computing the score.
        seed (int, optional): random seed.
    Returns:
        AbstractHeuristic: a Baal query strategy function
    """
    if name == "random":
        f_query_strategy = Random(
            shuffle_prop=shuffle_prop, reduction=reduction, seed=seed
        )
    elif name == "least_confidence":
        f_query_strategy = Certainty(shuffle_prop=shuffle_prop, reduction=reduction)

    elif name == "batch_bald":
        f_query_strategy = BatchBALD(num_samples=query_size, shuffle_prop=shuffle_prop)
    else:
        raise ValueError("{} is not an available f_query_strategy".format(name))

    return f_query_strategy


# Metrics
# ----------
def compute_metrics(
    y_true: list,
    y_pred: list,
    pos_label: int = 1,
    labels: Optional[list] = None,
    average: Optional[str] = None,
    sample_weight: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Computes the recall, precision and f1-score

    Args:
        y_true (list): true y values
        y_pred (list): predicted y values
        pos_label (int): Defaults to 1.
        labels (list, optional): possible labels
        average (str, optional):
        sample_weight (float, optional):

    Returns:
        Dict: precision, recall and F1-score

    """
    p, r, f1, support = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division="warn",
    )
    return {"p": p, "r": r, "f1": f1}


def compute_metrics_transformer(
    eval_preds: EvalPrediction,
    pos_label: int = 1,
    labels: Optional[list] = None,
    average: Optional[str] = None,
    sample_weight: Optional[float] = None,
) -> Dict[str, float]:
    """Computes the desired metrics given a evaluation prediction"""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    return compute_metrics(
        y_true=labels,
        y_pred=predictions,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
    )


# Tokenization 
# ------------
def tokenize_pairs(
    tokenizer: PreTrainedTokenizer, dataset: Dataset, max_seq_len: int = 256
) -> Dataset:
    """
    Tokenizes a dataset

    Args:
        tokenizer (PreTrainedTokenizer): HF tokenizer.
        dataset (Dataset): dataset to be tokenized.
        max_seq_len (int): maximum length of a sentence. Defaults to 256.

    Returns:
        Dataset: tokenized dataset
    """

    dataset = dataset.map(
        lambda e: tokenizer(
            text=e["sentence"],
            text_pair=e["text"],
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            truncation="longest_first",
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        ),
        batched=True,
        batch_size=32,
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "seq_length", "char_length", "label"],
    )
    return dataset


def tokenize(
    tokenizer: PreTrainedTokenizer, dataset: Dataset, max_seq_len: int = 256
) -> Dataset:
    """
    Tokenizes a dataset

    Args:
        tokenizer (PreTrainedTokenizer): HF tokenizer.
        dataset (Dataset): dataset to be tokenized.
        max_seq_len (int): maximum length of a sentence. Defaults to 256.

    Returns:
        Dataset: tokenized dataset
    """
    dataset = dataset.map(
        lambda e: tokenizer(
            e["sentence"],
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        ),
        batched=True,
        batch_size=32,
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "seq_length", "char_length", "label"],
    )
    return dataset
