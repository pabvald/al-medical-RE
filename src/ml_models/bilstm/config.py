# Base Dependencies
# -----------------
from dataclasses import dataclass


# Configuration Classes
# ---------------------
@dataclass
class LSTMConfig:
    hidden_size: int = 128
    emb_size: int = 256  # input size
    num_layers: int = 2
    dropout: float = 0.25
    emb_droput: float = 0.25 
    bidirectional: bool = True
    bias: bool = True
    batch_first: bool = True


@dataclass
class RDEmbeddingConfig:
    input_dim: int = 5
    embedding_dim: int = 5
    scale: bool = False
    freeze: bool = False


@dataclass
class EmbeddingConfig:
    embedding_dim: int = 5
    vocab_size: int = 2
    padding_idx: int = 0
    freeze: bool = True
    emb_path: str = ""
