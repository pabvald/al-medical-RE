# Base Dependencies
# -----------------
import numpy as np
from typing import List, Union, Any

# 3rd-Party Dependencies
# ----------------------
import torch

from baal.active import ActiveLearningDataset
from baal.active.dataset.base import Dataset
from datasets import Dataset as HFDataset


# Auxiliar Functions 
# ------------------
def my_active_huggingface_dataset(
    dataset,
    tokenizer=None,
    target_key: str = "label",
    input_key: str = "sentence",
    max_seq_len: int = 128,
    **kwargs
):
    """
    Wrapping huggingface.datasets with baal.active.ActiveLearningDataset.
    Args:
        dataset (torch.utils.data.Dataset): a dataset provided by huggingface.
        tokenizer (transformers.PreTrainedTokenizer): a tokenizer provided by huggingface.
        target_key (str): target key used in the dataset's dictionary.
        input_key (str): input key used in the dataset's dictionary.
        max_seq_len (int): max length of a sequence to be used for padding the shorter sequences.
        kwargs (Dict): Parameters forwarded to 'ActiveLearningDataset'.
    Returns:
        an baal.active.ActiveLearningDataset object.
    """

    return MyActiveLearningDatasetBert(
        MyHuggingFaceDatasets(dataset, tokenizer, target_key, input_key, max_seq_len),
        **kwargs
    )


# Datasets
# --------
class MyHuggingFaceDatasets(Dataset):
    """
    Support for `huggingface.datasets`: (https://github.com/huggingface/datasets).
    The purpose of this wrapper is to separate the labels from the rest of the sample information
    and make the dataset ready to be used by `baal.active.ActiveLearningDataset`.
    Args:
        dataset (Dataset): a dataset provided by huggingface.
        tokenizer (transformers.PreTrainedTokenizer): a tokenizer provided by huggingface.
        target_key (str): target key used in the dataset's dictionary.
        input_key (str): input key used in the dataset's dictionary.
        max_seq_len (int): max length of a sequence to be used for padding the shorter
            sequences.
    """

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer=None,
        target_key: str = "label",
        input_key: str = "sentence",
        max_seq_len: int = 128,
    ):
        self.dataset = dataset
        self.targets, self.texts = self.dataset[target_key], self.dataset[input_key]
        self.targets_list: List = np.unique(self.targets).tolist()

        if tokenizer:
            self.input_ids, self.attention_masks = self._tokenize(
                tokenizer, max_seq_len
            )
        else:
            self.input_ids = self.dataset["input_ids"]
            self.attention_masks = self.dataset["attention_mask"]

    @property
    def num_classes(self):
        return len(self.targets_list)

    def _tokenize(self, tokenizer, max_seq_len):
        # For speed purposes, we should use fast tokenizers here, but that is up to the caller
        tokenized = tokenizer(
            self.texts,
            add_special_tokens=True,
            max_length=max_seq_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def label(self, idx: int, value: int):
        """Label the item.
        Args:
            idx: index to label
            value: Value to label the index.
        """
        self.targets[idx] = value

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        target = self.targets_list.index(self.targets[idx])

        return {
            "input_ids": self.input_ids[idx].flatten()
            if len(self.input_ids) > 0
            else None,
            "inputs": self.texts[idx],
            "attention_mask": self.attention_masks[idx].flatten()
            if len(self.attention_masks) > 0
            else None,
            "label": torch.tensor(target, dtype=torch.long),
        }


class MyActiveLearningDatasetBert(ActiveLearningDataset):
    """
    MyActiveLearningDataset

    Modification of ActiveLearningDataset to allow the indexing with a
    a list of integers.
    """
    
    
    @property
    def labels(self) -> List[int]:
        return self._dataset[self.get_indices_for_active_step()]["label"]

    @property
    def n_labelled_tokens(self) -> int:
        return (
            self._dataset.dataset[self.get_indices_for_active_step()]["seq_length"]
            .sum()
            .item()
        )

    @property
    def n_labelled_chars(self) -> int:
        return (
            self._dataset.dataset[self.get_indices_for_active_step()]["char_length"]
            .sum()
            .item()
        )

    def __getitem__(self, index: Union[int, List[int]]) -> Any:
        """Return items from the original dataset based on the labelled index."""
        _index = np.array(self.get_indices_for_active_step())[index]
        return self._dataset[_index]


class MyActiveLearningDatasetBilstm(ActiveLearningDataset):
    """
    MyActiveLearningDataset

    Modification of ActiveLearningDataset to allow the indexing with a
    a list of integers.
    """

    @property
    def labels(self) -> List[int]:
        return self._dataset[self.get_indices_for_active_step()]["label"]

    @property
    def n_labelled_tokens(self) -> int:
        return (
            self._dataset[self.get_indices_for_active_step()]["seq_length"].sum().item()
        )

    @property
    def n_labelled_chars(self) -> int:
        return (
            self._dataset[self.get_indices_for_active_step()]["char_length"]
            .sum()
            .item()
        )

    def __getitem__(self, index: Union[int, List[int]]) -> Any:
        """Return items from the original dataset based on the labelled index."""
        _index = np.array(self.get_indices_for_active_step())[index]
        return self._dataset[_index]
