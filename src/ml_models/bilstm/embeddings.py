# coding: utf-8
"""
Embedding module

"""

# Base Dependencies
# -----------------
import math
import logging
from pathlib import Path
from typing import Dict, Optional

# Local Dependencies
# ------------------
from utils import freeze_params
from vocabulary import Vocabulary

# PyTorch Dependencies
# --------------------
import torch
from torch import Tensor, nn


logger = logging.getLogger(__name__)


# Embeddings Class
# ----------------
class Embeddings(nn.Module):
    """
    Simple embeddings class

    Source: https://github.com/joeynmt/joeynmt/blob/main/joeynmt/embeddings.py
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        scale: bool = False,
        vocab_size: int = 0,
        padding_idx: Optional[int] = 0,
        freeze: bool = False,
        **kwargs,
    ):
        """Creates a new embedding for the vocabulary

        Args:
            embedding_dim (int, optional): the embedding dimension. Defaults to 64.
            scale (bool, optional): indicates if the embeddings will be scale tiems the sqrt of their dimension. Defaults to False.
            vocab_size (int, optional): size of the vocabulary, i.e., input dimension. Defaults to 0.
            padding_idx (int, optional): index used for padding. Defaults to 1.
            freeze (bool, optional): indicates if the embeddings are trained (False) or left untouched (True). Defaults to False.
        """
        # pylint: disable=unused-argument
        super().__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor) -> Tensor:
        """Perform lookup for input `x` in the embedding table.

        Args:
            x (Tensor): index in the vocabulary
        Returns:
            embedded representation for `x`
        """

        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"vocab_size={self.vocab_size})"
        )

    # from fairseq
    def load_from_file(self, embed_path: Path, vocab: Vocabulary) -> None:
        """Loads pretrained embedding weights from text file.
        - First line is expected to contain vocabulary size and dimension.
          The dimension has to match the model's specified embedding size,
          the vocabulary size is used in logging only.
        - Each line should contain word and embedding weights
          separated by spaces.
        - The pretrained vocabulary items that are not part of the
          joeynmt's vocabulary will be ignored (not loaded from the file).
        - The initialization (specified in config["model"]["embed_initializer"])
          of joeynmt's vocabulary items that are not part of the
          pretrained vocabulary will be kept (not overwritten in this func).
        - This function should be called after initialization!
        Example:
            2 5
            the -0.0230 -0.0264  0.0287  0.0171  0.1403
            at -0.0395 -0.1286  0.0275  0.0254 -0.0932

        Args:
            embed_path (Path): embedding weights text file
            vocab (Vocabulary): Vocabulary object
        """
        # pylint: disable=logging-too-many-args
        unk_in = False
        bos_in = False
        eos_in = False
        embed_dict: Dict[int, Tensor] = {}
        # parse file
        with embed_path.open("r", encoding="utf-8", errors="ignore") as f_embed:
            vocab_size, d = map(int, f_embed.readline().split())
            assert self.embedding_dim == d, "Embedding dimension doesn't match."
            for line in f_embed.readlines():
                tokens = line.rstrip().split(" ")
                if tokens[0] in vocab.specials or not vocab.is_unk(tokens[0]):
                    if vocab.lookup(tokens[0]) == vocab.unk_index:
                        unk_in = True
                    # elif vocab.lookup(tokens[0]) == vocab.bos_index:
                    #     bos_in = True
                    # elif vocab.lookup(tokens[0]) == vocab.eos_index:
                    #     eos_in = True

                    embed_dict[vocab.lookup(tokens[0])] = torch.FloatTensor(
                        [float(t) for t in tokens[1:]]
                    )

            logger.warning(
                "Loaded {} of {} ({}) tokens om the pretrained WE.".format(
                    len(embed_dict),
                    len(vocab),
                    len(embed_dict) / len(vocab),
                )
            )

        # assign
        for idx, weights in embed_dict.items():
            if idx < self.vocab_size:
                assert self.embedding_dim == len(weights)
                self.lut.weight.data[idx] = weights

        if not unk_in:
            self.lut.weight.data[vocab.unk_index] = torch.mean(
                self.lut.weight.data, axis=0
            )

        logger.warning(
            "Loaded {} of {} ({}) tokens of the vocabulary.".format(
                len(embed_dict),
                len(vocab),
                len(embed_dict) / len(vocab),
            )
        )


# RDEmbeddings Class
# -----------------
class RDEmbeddings(Embeddings):
    def __init__(
        self,
        input_dim: int = 0,
        embedding_dim: int = 64,
        scale: bool = False,
        freeze: bool = False,
        **kwargs,
    ):
        """Relative Distance Embedding

        Args:
            input_dim (int, optional): the maximum absolute value of positions. Defaults to 0.
            embedding_dim (int, optional): the embedding dimension. Defaults to 64.
            scale (bool, optional): indicates if the embeddings will be scale tiems the sqrt of their dimension. Defaults to False.
            freeze (bool, optional): indicates if the embeddings are trained (False) or left untouched (True). Defaults to False.
        """
        self.input_dim = input_dim

        super().__init__(
            embedding_dim=embedding_dim,
            scale=scale,
            vocab_size=(self.input_dim * 2 + 1),
            padding_idx=None,
            freeze=freeze,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform lookup for input `x` in the embedding table.

        Args:
            x (Tensor): index in the vocabulary
        Returns:
            embedded representation for `x`
        """

        # delimits relative distance values to the input dimension
        x = torch.clamp(x, min=-self.input_dim, max=self.input_dim) + self.input_dim

        return super().forward(x)
