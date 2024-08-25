# Base Dependencies
# -----------------
import numpy as np
from typing import Dict, List

# PyTorch Dependencies
# ---------------------
import torch
from torch import Tensor


# Auxiliar Functions
# -------------------
def sort_batch(
    batch: Dict[str, List[List[float]]], lengths: List[List[float]]
) -> Dict[str, List[List[float]]]:
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths. This way the output can be used by pack_padded_sequences(...)

    Args:
        batch (Dict[str,  List[List[float]]]): batch of data

    Return:
        Dict[str,  List[List[float]]]: batch of data ordered in descending order of sequence length.

    """
    perm_idx = np.argsort(-lengths)

    for key in batch.keys():
        batch[key] = batch[key][perm_idx]

    return batch


def pad_seqs(
    seqs: List[List[float]], lengths: List[int], padding_idx: int
) -> List[List[float]]:
    """Pads sequences

    Args:
        seqs (List[List[float]]): sequences of different lengths
        lengths (List[int]): length of each sequence
        padding_idx (int): value used for padding

    Returns:
        List[List[float]]: padded sequences
    """
    batch_size = len(lengths)
    max_length = max(lengths)

    padded_seqs = np.full(
        shape=(batch_size, max_length), fill_value=padding_idx, dtype=np.int32
    )

    for i, l in enumerate(lengths):
        padded_seqs[i, 0:l] = seqs[i]

    return padded_seqs


def pad_and_sort_batch(batch: Dict, padding_idx: int, rd_max: int) -> Dict[str, Tensor]:
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    for key in ["char_length", "seq_length", "label"]:
        batch[key] = np.array(batch[key])

    for key in ["e1", "e2"]:
        seqs = batch[key]
        # pad entities apart to avoid unnecessary padding
        lengths = list(map(lambda x: len(x), seqs))
        batch[key] = pad_seqs(seqs, lengths, padding_idx)

    for key in ["rd1", "rd2"]:
        seqs = batch[key]
        # pad relative distance with maximum value
        batch[key] = pad_seqs(seqs, batch["seq_length"], rd_max)

    for key in ["sent", "iob", "pos", "dep"]:
        seqs = batch[key]
        # pad other features with the common padding index
        batch[key] = pad_seqs(seqs, batch["seq_length"], padding_idx)

    return sort_batch(batch, batch["seq_length"])


def custom_collate(data: Dict[str, List[List[float]]]):
    """Separates the inputs and the targets

    Args:
        data (Dict[str, List[List[float]]]): batch of data

    Returns:
        Tuple[Dict[str, Tensor], Tensor]: inputs and targets.

    """
    inputs = {}
    targets = torch.from_numpy(data[0]["label"]).long()
    for key, value in data[0].items():
        if key != "label":
            inputs[key] = torch.from_numpy(value).long()
    return inputs, targets
