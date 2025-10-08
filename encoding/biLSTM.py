
###################
# BiLSTM Encoder  #
###################
# This module defines a Bidirectional LSTM encoder for processing sequences.
# It takes embedded sequences and their lengths as input and outputs the final hidden states.
# The final hidden states from both directions are concatenated and passed through a dropout layer.

#connect output to logistic regression layer
###################

import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BiLSTMEncoder(nn.Module):
    
    def __init__(self, 
                embedding_dim: int,
                hidden_dim: int,
                num_layers: int = 2,
                dropout: float = 0.3,
                bidirectional: bool = True):
        """
        Initialize BiLSTM encoder.
        
        Args:
            embedding_dim: Dimension of input word embeddings (from Word2Vec)
            hidden_dim: Hidden state dimension for LSTM
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BiLSTMEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer for output
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        logger.info(f"BiLSTM Encoder initialized: embedding_dim={embedding_dim}, "
                f"hidden_dim={hidden_dim}, num_layers={num_layers}, "
                f"bidirectional={bidirectional}")
    
    def forward(self, embedded_sequences, lengths):
        """
        Forward pass through BiLSTM.
        
        Args:
            embedded_sequences: [batch_size, seq_len, embedding_dim] Word embeddings from Word2Vec
            lengths: [batch_size] Actual lengths of sequences (for packing)
        
        Returns:
            final_hidden: [batch_size, hidden_dim * num_directions] Concatenated final hidden states from both directions
        """
        batch_size = embedded_sequences.size(0)
        
        # Pack padded sequences for efficient computation
        # This ignores padding tokens during LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embedded_sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        # packed_output: packed sequence of all hidden states
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # cell: [num_layers * num_directions, batch_size, hidden_dim]
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack the output sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Extract final hidden states from last layer
        if self.bidirectional:
            # hidden[-2]: forward direction from last layer
            # hidden[-1]: backward direction from last layer
            forward_hidden = hidden[-2, :, :]   # [batch_size, hidden_dim]
            backward_hidden = hidden[-1, :, :]  # [batch_size, hidden_dim]
            
            # Concatenate forward and backward hidden states
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
            # final_hidden: [batch_size, hidden_dim * 2]
        else:
            # Only forward direction
            final_hidden = hidden[-1, :, :]  # [batch_size, hidden_dim]
        
        # Apply layer normalization
        final_hidden = self.layer_norm(final_hidden)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        return final_hidden
    
    def get_output_dim(self):
        """Returns the output dimension of the encoder."""
        return self.hidden_dim * self.num_directions



####################
# Testing BiLSTM Encoder #
####################
# Uncomment below to test the BiLSTMEncoder independently

if __name__ == "__main__":
    """Example of using BiLSTM encoder."""
    print("=== BiLSTM Encoder Example ===")
    
    # Parameters
    batch_size = 16
    seq_len = 50
    embedding_dim = 100  # Word2Vec dimension
    hidden_dim = 128
    
    # Create encoder
    encoder = BiLSTMEncoder(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3
    )
    
    # Dummy input (word embeddings from Word2Vec)
    embedded_sequences = torch.randn(batch_size, seq_len, embedding_dim)
    lengths = torch.randint(20, seq_len, (batch_size,))
    
    # Forward pass
    output = encoder(embedded_sequences, lengths)
    print(f"Input shape: {embedded_sequences.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dimension: {encoder.get_output_dim()}")

###################