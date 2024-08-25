# Base Dependencies
# -----------------
import functools
import numpy as np
import operator
import os
import random
import re

from glob import glob
from os.path import join as pjoin
from pathlib import Path
from typing import List, Any, Union

# Local Dependencies
# ------------------
from constants import N2C2_PATH, DDI_PATH, N2C2_ANNONYM_PATTERNS, DDI_ALL_TYPES

# 3rd-Party Dependencies
# ----------------------
import torch
from torch import nn
from transformers import set_seed as transformers_set_seed


def set_seed(seed: int) -> None:
    """Sets the random seed for modules torch, numpy and random.

    Args:
        seed (int): random seed
    """
    transformers_set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def flatten(array: List[List[Any]]) -> List[Any]:
    """
    Flattens a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().

    Args:
        array (List[List[Any]]): a nested list
    Returns:
        List[Any]: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])


def write_list_to_file(output_path: Path, array: List[Any]) -> None:
    """
    Writes list of str to file in `output_path`.

    Args:
        output_path (Path): output file path
        array (List[Any]): list of strings
    """
    with output_path.open("w", encoding="utf-8") as opened_file:
        for entry in array:
            opened_file.write(f"{entry}\n")


def read_list_from_file(input_path: Path) -> List[str]:
    """
    Reads list of str from file in `input_path`.

    Args:
        input_path (Path): input file path
    Returns:
        List[str]: list of strings
    """
    if input_path is None:
        return []

    tokens = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        tokens.append(line.rstrip("\n"))

    return tokens


def make_dir(dirpath: str):
    """Creates a directory if it doesn't exist"""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def freeze_params(module: nn.Module) -> None:
    """
    Freezes the parameters of this module,
    i.e. do not update them during training

    Args:
        module (nn.Module): freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def ddi_binary_relation(rel_type: Union[str, int]) -> int:
    """Converts a DDI's relation type into binary

    Args:
        rel_type (str): relation type

    Returns:
        int: 0 if the relation type is `"NO-REL"`, `"0"` or `0`,
        1 if the relation type is an string in `["EFFECT", "MECHANISM", "ADVISE", "INT"]` or is an integer `> 0`.
    """

    rt = rel_type
    if isinstance(rt, str):
        if rt in DDI_ALL_TYPES:
            rt = DDI_ALL_TYPES.index(rt)
        else:
            rt = int(rt)
    if rt == 0:
        return 0
    else:
        return 1


def doc_id_n2c2(filepath: str) -> str:
    """Extracts the document id of a n2c2 filepath"""
    return re.findall(r"\d{2,}", filepath)[-1]


def doc_id_ddi(filepath: str) -> str:
    """Extracts the document id of a ddi filepath"""
    file_name = filepath.split()[-1]
    return file_name[:-4]


def clean_text_ddi(text: str) -> str:
    """Cleans text of a text fragment from a ddi document

    Args:
        text (str): text fragment

    Returns:
        str: cleaned text fragment
    """
    # remove more than one space
    text = re.sub(r"[\s]+", " ", text)

    # include space after ;
    text = re.sub(r";", "; ", text)

    return text


def clean_text_n2c2(text: str) -> str:
    """Cleans text of a text fragment from a n2c2 document

    Args:
        text (str): text fragment

    Returns:
        str: cleaned text fragment
    """

    # remove head and tail spaces
    # text = text.strip()

    # remove newlines
    text = re.sub(r"\n", " ", text)

    # substitute annonymizations by their type
    for repl, pattern in N2C2_ANNONYM_PATTERNS.items():
        text = re.sub(pattern, repl, text)

    # remove not matching annonymizations
    text = re.sub(r"\[\*\*[^\*]+\*\*\]", "", text)

    # remove more than one space
    text = re.sub(r"[\s]+", " ", text)

    # replace two points by one
    text = re.sub(r"\.\.", ".", text)

    return text


def files_n2c2():
    """Loads the filepaths of the n2c2 dataset splits"""
    splits = {}

    for split in ["train", "test"]:
        files = glob(pjoin(N2C2_PATH, split, "*.txt"))
        splits[split] = list(map(lambda file: file[:-4], files))

    return splits


def files_ddi():
    """Loads the filepaths of the DDI corpus splits"""
    splits = {}

    for split in ["train", "test"]:
        if split == "train":
            splits["train"] = glob(pjoin(DDI_PATH, split, "DrugBank", "*.xml")) + glob(
                pjoin(DDI_PATH, split, "MedLine", "*.xml")
            )

        else:
            splits["test"] = glob(
                pjoin(DDI_PATH, split, "re", "DrugBank", "*.xml")
            ) + glob(pjoin(DDI_PATH, split, "re", "MedLine", "*.xml"))

    return splits
