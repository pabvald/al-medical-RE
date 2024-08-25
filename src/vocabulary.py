# coding: utf-8
"""
Vocabulary module

Source: https://github.com/joeynmt/joeynmt/blob/main/joeynmt/vocabulary.py
"""

# Base Dependencies
# -----------------
import sys
import logging
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Local Dependencies
# ------------------
from constants import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
)
from models.relation_collection import RelationCollection
from utils import read_list_from_file, write_list_to_file

# Constants
# ---------
from constants import DATASETS_PATHS, N2C2_VOCAB_PATH, DDI_VOCAB_PATH

VOC_MIN_FREQ = 10


logger = logging.getLogger(__name__)


class Vocabulary:
    """Vocabulary represents mapping between tokens and indices."""

    def __init__(self, tokens: List[str]) -> None:
        """
        Create vocabulary from list of tokens.
        Special tokens are added if not already in list.

        Args:
            tokens (List[str]): list of tokens
        """
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        self.specials = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # don't allow to access _stoi and _itos outside of this class
        self._stoi: Dict[str, int] = {}  # string to index
        self._itos: List[str] = []  # index to string

        # construct
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign after stoi is built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        self.unk_index = self.lookup(UNK_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self.unk_index == UNK_ID
        assert self._itos[UNK_ID] == UNK_TOKEN

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary

        Args:
            tokens (List[str]): list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self._itos)
            # add to vocab if not already there
            if t not in self._itos:
                self._itos.append(t)
                self._stoi[t] = new_index

    def to_file(self, file: Path) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        Args:
            file (Path): path to file where the vocabulary is written
        """
        write_list_to_file(file, self._itos)

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        Args:
            token (str):
        Returns:
            bool: True if covered, False otherwise
        """
        return self.lookup(token) == UNK_ID

    def lookup(self, token: str) -> int:
        """
        look up the encoding dictionary. (needed for multiprocessing)

        Args:
            token (str): surface str
        Returns:
            int: token id
        """
        return self._stoi.get(token, UNK_ID)

    def __len__(self) -> int:
        return len(self._itos)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vocabulary):
            return self._itos == other._itos
        return False

    def array_to_sentence(
        self, array: np.ndarray, cut_at_eos: bool = True, skip_pad: bool = True
    ) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result off at the
        end-of-sequence token.

        Args:
            array (numpy.ndarray): 1D array containing indices
            cut_at_eos (bool): cut the decoded sentences at the first <eos>
            skip_pad (bool): skip generated <pad> tokens

        Returns:
            List[str]: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self._itos[i]
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
            # break at the position AFTER eos
            if cut_at_eos and s == EOS_TOKEN:
                break
        return sentence

    def arrays_to_sentences(
        self, arrays: np.ndarray, cut_at_eos: bool = True, skip_pad: bool = True
    ) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their sentences,
        optionally cutting them off at the end-of-sequence token.

        Args:
            arrays (numpy.ndarray): 2D array containing indices
            cut_at_eos (bool): cut the decoded sentences at the first <eos>
            skip_pad (bool): skip generated <pad> tokens
        Returns:
            List[List[str]]: list of list of strings (tokens)
        """
        return [
            self.array_to_sentence(
                array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad
            )
            for array in arrays
        ]

    def sentences_to_ids(
        self,
        sentences: List[List[str]],
        padded: bool = False,
        bos: bool = False,
        eos: bool = False,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode sentences to indices and pad sequences to the maximum length of the
        sentences given if necessary

        Args:
            sentences List[List[str]]: list of tokenized sentences

        Returns:
            - padded ids
            - original lengths before padding
        """
        max_len = max([len(sent) for sent in sentences])
        if bos:
            max_len += 1
        if eos:
            max_len += 1
        sentences_enc, lengths = [], []
        for sent in sentences:
            encoded = [self.lookup(s) for s in sent]
            if bos:
                encoded = [self.bos_index] + encoded
            if eos:
                encoded = encoded + [self.eos_index]
            if padded:
                offset = max(0, max_len - len(encoded))
                sentences_enc.append(encoded + [self.pad_index] * offset)
            else:
                sentences_enc.append(encoded)
            lengths.append(len(encoded))
        return sentences_enc, lengths

    def log_vocab(self, k: int) -> str:
        """first k vocab entities"""
        return " ".join(f"({i}) {t}" for i, t in enumerate(self._itos[:k]))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(len={self.__len__()}, "
            f"specials={self.specials})"
        )

    @staticmethod
    def sort_and_cut(
        counter: Counter, max_size: int = sys.maxsize, min_freq: int = -1
    ) -> List[str]:
        """
        Cut counter to most frequent, sorted numerically and alphabetically

        Args:
            counter (Counter): flattened token list in Counter object
            max_size (int): maximum size of vocabulary
            min_freq (int): minimum frequency for an item to be included

        Returns:
            List[str]: valid tokens
        """
        # filter counter by min frequency
        if min_freq > -1:
            counter = Counter({t: c for t, c in counter.items() if c >= min_freq})

        # sort by frequency, then alphabetically
        tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # cut off
        vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
        assert len(vocab_tokens) <= max_size, (len(vocab_tokens), max_size)
        return vocab_tokens

    @staticmethod
    def build_vocab(
        cfg: Dict, collection: Optional[RelationCollection] = None
    ) -> "Vocabulary":
        """
        Builds vocabulary either from file or sentences.

        Args:
            cfg (Dict): data cfg

        Returns:
            Vocabulary: created from either `tokens` or `vocab_file`
        """
        vocab_file = cfg.get("voc_file", None)
        min_freq = cfg.get("voc_min_freq", 1)  # min freq for an item to be included
        max_size = int(cfg.get("voc_limit", sys.maxsize))  # max size of vocabulary
        assert max_size > 0

        if vocab_file is not None:
            # load it from file (not to apply `sort_and_cut()`)
            unique_tokens = read_list_from_file(Path(vocab_file))

        elif collection is not None:
            # tokenize sentences
            tokens = []
            for doc in collection.tokens:
                for t in doc:
                    tokens.append(t.text.lower())

            # newly create unique token list (language-wise)
            counter = Counter(tokens)
            unique_tokens = Vocabulary.sort_and_cut(counter, max_size, min_freq)
        else:
            raise Exception("Please provide a vocab file path or a relation collection.")

        vocab = Vocabulary(unique_tokens)
        assert len(vocab) <= max_size + len(vocab.specials), (len(vocab), max_size)

        # check for all except for UNK token whether they are OOVs
        for s in vocab.specials:
            assert s == UNK_TOKEN or not vocab.is_unk(s)

        return vocab

    @staticmethod
    def create_vocabulary(dataset: str, train_collection: RelationCollection, save_to_disk: bool = True) -> "Vocabulary":
        """Creates the vocabulary of a dataset

        Args:
            dataset (str): dataset's name
            train_collection (RelationCollection): train split of the dataset

        Returns:
            Vocabulary: _description_
        """
        # configuration
        cfg = {
            "voc_min_freq": VOC_MIN_FREQ,
        }
        # create vocabulary
        vocabulary = Vocabulary.build_vocab(cfg=cfg, collection=train_collection)
        print(
            "Vocabulary created for {} dataset: {} tokens".format(dataset, len(vocabulary))
        )

        # save vocab to file
        if save_to_disk:
            vocab_file = DATASETS_PATHS[dataset]
            vocabulary.to_file(vocab_file)

        return vocabulary


    def load_vocab(dataset: str) -> "Vocabulary":
        """Loads the vocabulary of a dataset

        Args:
            dataset (str): dataset's name

        Returns:
            Vocabulary: vocabulary of the dataset
        """
        path = {"n2c2": N2C2_VOCAB_PATH, "DDI": DDI_VOCAB_PATH}[dataset]

        return Vocabulary(read_list_from_file(path))