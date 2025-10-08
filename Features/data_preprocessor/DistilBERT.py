
###################
# DistilBERT Encoder  #
###################
# This module defines a DistilBERT encoder for processing sequences.
# It takes tokenized sequences and attention masks as input and outputs the pooled representation.
# The pooled representation is obtained using a specified pooling strategy (CLS, mean, or max).
# The output is passed through a dropout layer and layer normalization for stability.

#connect the output to logistic regression layer
###################

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DistilBERTEncoder(nn.Module):
    """
    DistilBERT encoder for contextual text encoding.
    Uses pretrained transformer model with optional fine-tuning.
    """
    
    def __init__(self,
                model_name: str = 'distilbert-base-uncased',
                freeze_bert: bool = False,
                dropout: float = 0.3,
                pooling_strategy: str = 'cls'):
        """
        Initialize DistilBERT encoder.
        
        Args:
            model_name: Pretrained DistilBERT model name
            freeze_bert: Whether to freeze DistilBERT parameters
            dropout: Dropout probability for output
            pooling_strategy: How to pool token embeddings
                            'cls': Use [CLS] token
                            'mean': Mean pooling of all tokens
                            'max': Max pooling of all tokens
        """
        super(DistilBERTEncoder, self).__init__()
        
        self.pooling_strategy = pooling_strategy
        
        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze parameters if specified
        if freeze_bert:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            logger.info("DistilBERT parameters frozen")
        else:
            logger.info("DistilBERT parameters will be fine-tuned")
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(768)  # DistilBERT hidden size
        
        logger.info(f"DistilBERT Encoder initialized: model={model_name}, "
                f"freeze={freeze_bert}, pooling={pooling_strategy}")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through DistilBERT.
        
        Args:
            input_ids: [batch_size, seq_len] Token IDs from tokenizer
            attention_mask: [batch_size, seq_len] Attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            pooled_output: [batch_size, 768] Pooled representation
        """
        # DistilBERT forward pass
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        # last_hidden_state: [batch_size, seq_len, 768]
        last_hidden_state = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            pooled_output = last_hidden_state[:, 0, :]
            
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            # Expand mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Sum of masked hidden states
            sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)
            
            # Divide by number of non-padding tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
            
        elif self.pooling_strategy == 'max':
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Set padding positions to very negative number
            last_hidden_state = last_hidden_state.clone()
            last_hidden_state[mask_expanded == 0] = -1e9
            
            # Max pool
            pooled_output = torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Apply layer normalization
        pooled_output = self.layer_norm(pooled_output)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output
    
    def get_output_dim(self):
        """Returns the output dimension of the encoder."""
        return 768  # DistilBERT hidden size
