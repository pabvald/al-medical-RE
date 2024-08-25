# coding: utf-8
"""
RNN encoders

Source: https://github.com/joeynmt/joeynmt/blob/main/joeynmt/encoders.py
"""

# Base Dependencies
# -----------------
from typing import Tuple

# Local Dependencies
# -------------------
from utils import freeze_params


# PyTorch Dependencies
# --------------------
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """
    Base encoder class
    """

    # pylint: disable=abstract-method
    @property
    def output_size(self):
        """
        Returns the output size
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """Encodes a sequence of word embeddings"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs,
    ) -> None:
        """Create a new recurrent encoder.

        Args:
            rnn_type (str): RNN type: `gru` or `lstm`.
            hidden_size (int): Size of each RNN.
            emb_size (int): Size of the word embeddings.
            num_layers (int): Number of encoder RNN layers.
            dropout (float):  Is applied between RNN layers.
            emb_dropout (float): Is applied to the RNN input (word embeddings).
            bidirectional (bool): Use a bi-directional RNN.
            freeze (bool): freeze the parameters of the encoder during training
            kwargs:
        """
        super().__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor
    ) -> None:
        """
        Make sure the shape of the inputs to `self.forward` are correct.
        Same input semantics as `self.forward`.

        Args:
            embed_src (Tensor): embedded source tokens
            src_length (Tensor): source length
        """
        # pylint: disable=unused-argument
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        assert len(src_length.shape) == 1

    def forward(
        self, embed_src: Tensor, src_length: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.

        Args:
            embed_src: embedded src inputs, shape (batch_size, src_len, embed_size)
            src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
            kwargs:

        Returns:
            output: hidden states with shape (batch_size, max_length, directions*hidden),
            hidden_concat: last hidden state with shape (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(embed_src=embed_src, src_length=src_length)
        total_length = embed_src.size(1)

        # apply dropout to the rnn input
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(
            embed_src, src_length.cpu(), batch_first=True, enforce_sorted=True
        )
        output, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden  # pylint: disable=unused-variable

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=total_length
        )
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        batch_size = hidden.size()[1]
        # separate final hidden states by layer and direction
        hidden_layerwise = hidden.view(
            self.rnn.num_layers,
            2 if self.rnn.bidirectional else 1,
            batch_size,
            self.rnn.hidden_size,
        )
        # final_layers: layers x directions x batch x hidden

        # concatenate the final states of the last layer for each directions
        # thanks to pack_padded_sequence final states don't include padding
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]

        # only feed the final state of the top-most layer to the decoder
        # pylint: disable=no-member
        hidden_concat = torch.cat([fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)
        # final: batch x directions*hidden

        assert hidden_concat.size(0) == output.size(0), (
            hidden_concat.size(),
            output.size(),
        )
        return output, hidden_concat

    def __repr__(self):
        return f"{self.__class__.__name__}(rnn={self.rnn})"
