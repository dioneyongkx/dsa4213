import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        embedding_matrix,          # np.ndarray [vocab_size, embed_dim]
        pad_id: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        freeze_embeddings: bool = True,
    ):
        """
        BiLSTM Encoder that uses pretrained Word2Vec embeddings (SentencePiece ID → vector).
        Args:
            embedding_matrix: numpy array containing pretrained embeddings.
            pad_id: index used for <pad> token.
            hidden_dim: hidden size for the LSTM.
            num_layers: number of stacked LSTM layers.
            dropout: dropout between layers.
            bidirectional: if True, use a bidirectional LSTM.
            freeze_embeddings: if True, embeddings are frozen (not trained).
        """
        super(BiLSTMEncoder, self).__init__()

        num_embeddings, embed_dim = embedding_matrix.shape
        self.pad_id = pad_id
        self.embedding_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # --- Embedding layer ---
        self.emb = nn.Embedding(num_embeddings, embed_dim, padding_idx=pad_id)
        with torch.no_grad():
            self.emb.weight.copy_(torch.from_numpy(embedding_matrix))
        self.emb.weight.requires_grad = not freeze_embeddings  # keep frozen

        # --- LSTM layer ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Regularization ---
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        logger.info(
            f"BiLSTM Encoder initialized | emb_dim={embed_dim}, hidden_dim={hidden_dim}, "
            f"layers={num_layers}, bidirectional={bidirectional}, freeze_embeddings={freeze_embeddings}"
        )

    def forward(self, input_ids):
        """
        Forward pass through BiLSTM.
        Args:
            input_ids: [batch_size, seq_len] padded SentencePiece IDs
        Returns:
            final_hidden: [batch_size, hidden_dim * num_directions]
        """
        # Compute true sequence lengths (ignore padding)
        lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1)

        # Lookup embeddings
        embedded_sequences = self.emb(input_ids)  # [B, T, E]

        # Pack padded sequences for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Run LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Extract final hidden state
        if self.bidirectional:
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden[-1, :, :]

        # Normalize + dropout
        final_hidden = self.layer_norm(final_hidden)
        final_hidden = self.dropout(final_hidden)
        return final_hidden

    def get_output_dim(self):
        return self.hidden_dim * self.num_directions



# Full classifier (BiLSTM to Logistic Regression)

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        pad_id: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 2,
    ):
        """
        End-to-end spam/ham classifier using pretrained embeddings + BiLSTM + logistic regression head.
        Only LSTM and classifier weights are trained.
        """
        super(BiLSTMClassifier, self).__init__()

        # lstm
        self.encoder = BiLSTMEncoder(
            embedding_matrix=embedding_matrix,
            pad_id=pad_id,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            freeze_embeddings=True,  # freeze embedding
        )

        # log reg head
        self.classifier = nn.Linear(self.encoder.get_output_dim(), num_classes)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch, seq_len] padded SP IDs
        Returns:
            logits: [batch, num_classes]
        """
        feats = self.encoder(input_ids)    # [B, hidden*dirs]
        logits = self.classifier(feats)    # [B, 2]
        return logits