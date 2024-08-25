# Base Dependencies
# -----------------
from typing import Dict

# Package Dependencies
# --------------------
from .embeddings import Embeddings, RDEmbeddings
from .encoders import RecurrentEncoder
from .config import EmbeddingConfig, RDEmbeddingConfig, LSTMConfig

# Local Dependencies
# ------------------
from vocabulary import Vocabulary

# PyTorch Dependencies
# ---------------------
from torch import nn, Tensor, concat, mean


# Model
# -----
class HasanModel(nn.Module):
    """
    Implementation of the BiLSTM model described in `Hasan et al. (2020) - Integrating
    Text Embedding with Traditional NLP Features for Clinical Relation Extraction`
    """

    def __init__(
        self,
        vocab: Vocabulary,
        lstm_config: LSTMConfig,
        bioword2vec_config: EmbeddingConfig,
        rd_config: RDEmbeddingConfig,
        pos_config: EmbeddingConfig,
        dep_config: EmbeddingConfig,
        iob_config: EmbeddingConfig,
        num_classes: int = 2,
        clf_dropout: float = 0.25,
    ):
        """Initializes the model

        Args:
            vocab (Vocabulary): vocabulary object
            lstm_config (LSTMConfig): configuration for the LSTM encoder
            bioword2vec_config (EmbeddingConfig): configuration for the BioWord2Vec embedding
            rd_config (RDEmbeddingConfig): configuration for the Relative Distance embedding
            pos_config (EmbeddingConfig): configuration for the POS embedding
            dep_config (EmbeddingConfig): configuration for the DEP embedding
            iob_config (EmbeddingConfig): configuration for the IOB embedding
            num_classes (int, optional): number of output classes. Defaults to 2.
            clf_dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super(HasanModel, self).__init__()

        # attributes
        self.vocab = vocab
        self.lstm_config = lstm_config
        self.bioword2vec_config = bioword2vec_config
        self.rd_config = rd_config
        self.pos_config = pos_config
        self.dep_config = dep_config
        self.iob_config = iob_config
        self.num_classes = num_classes
        self.num_directions = 2 if self.lstm_config.bidirectional else 1
        self.clf_hidden_dim = 64
        self.clf_dropout = clf_dropout

        # embedding layers
        self.wv_embedding = Embeddings(**self.bioword2vec_config.__dict__)
        self.rd_embedding = RDEmbeddings(**self.rd_config.__dict__)
        self.pos_embedding = Embeddings(**self.pos_config.__dict__)
        self.dep_embedding = Embeddings(**self.dep_config.__dict__)
        self.iob_embedding = Embeddings(**self.iob_config.__dict__)

        # BiLSTM encoder
        self.lstm = RecurrentEncoder(rnn_type="lstm", **self.lstm_config.__dict__)

        # classifier
        self.fc = nn.Sequential(
            nn.Dropout(p=self.clf_dropout),
            nn.Linear(self.clf_input_dim, self.clf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.clf_dropout),
            nn.Linear(self.clf_hidden_dim, self.num_classes),
            nn.ReLU(),
            nn.Sigmoid(),
        )

        # load pretrained embeddings
        self.wv_embedding.load_from_file(self.bioword2vec_config.emb_path, self.vocab)

    @property
    def clf_input_dim(self) -> int:
        """Input dimensions of the classifier"""
        return (self.num_directions * self.lstm_config.hidden_size) + (
            2 * self.wv_embedding.embedding_dim
        )

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass of the model

        Args:
            inputs (Dict[str, Tensor]): input tensors

        Returns:
            Tensor: output tensor
        """
        e1: Tensor = inputs["e1"]  # [batch_size, max_len_e1]
        e2: Tensor = inputs["e2"]  # [batch_size, max_len_e2]
        sent: Tensor = inputs["sent"]  # [batch_size, max_len_seq]
        rd1: Tensor = inputs["rd1"]  # [batch_size, max_len_seq]
        rd2: Tensor = inputs["rd2"]  # [batch_size, max_len_seq]
        pos: Tensor = inputs["pos"]  # [batch_size, max_len_seq]
        dep: Tensor = inputs["dep"]  # [batch_size, max_len_seq]
        iob: Tensor = inputs["iob"]  # [batch_size, max_len_seq]
        seq_length: Tensor = inputs["seq_length"]  # [batch_size]

        assert len(e1.shape) == 2
        assert len(e2.shape) == 2
        assert len(sent.shape) == 2
        assert len(rd1.shape) == 2
        assert len(rd2.shape) == 2
        assert len(pos.shape) == 2
        assert len(dep.shape) == 2 
        assert len(iob.shape) == 2
        assert len(seq_length.shape) == 1

        # embedded inputs
        e1_emb = mean(self.wv_embedding(e1), axis=1)  # [batch_size, wv_emb_dim]
        e2_emb = mean(self.wv_embedding(e2), axis=1)  # [batch_size, wv_emb_dim]
        sent_emb = self.wv_embedding(sent)  # [batch_size, seq_length, wv_emb_dim]
        rd1_emb = self.rd_embedding(rd1)  # [batch_size, seq_length, rd_emb_dim]
        rd2_emb = self.rd_embedding(rd2)  # [batch_size, seq_length, rd_emb_dim]
        pos_emb = self.pos_embedding(pos)  # [batch_size, seq_length, pos_emb_dim]
        dep_emb = self.dep_embedding(dep)  # [batch_size, seq_length, pos_emb_dim]
        iob_emb = self.iob_embedding(iob)  # [batch_size, seq_length, iob_emb_dim]

        # encode
        inputs_emb = concat((sent_emb, rd1_emb, rd2_emb, pos_emb, dep_emb, iob_emb), axis=2)
        outputs_emb, hidden_concat = self.lstm(inputs_emb, seq_length)
        outputs = concat((e1_emb, e2_emb, hidden_concat), axis=1)

        # classify
        outputs = self.fc(outputs)

        return outputs
