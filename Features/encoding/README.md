# Email Scam Detection - Encoding Layer Documentation

## Overview

This module implements the encoding layer for email scam detection as part of the DSA4213 Final Project. The encoding layer is responsible for transforming preprocessed email text into fixed-dimensional vector representations that capture semantic and contextual information.

Two distinct encoder architectures are implemented:

- **BiLSTM Encoder**: A bidirectional Long Short-Term Memory network that processes Word2Vec embeddings
- **DistilBERT Encoder**: A pretrained transformer-based encoder using DistilBERT

These encoders serve as the core feature extraction component in the scam detection pipeline, bridging the gap between raw text input and the final classification layer.

## Architecture

### Pipeline A: Word2Vec → BiLSTM Encoder → Classifier

```
Input Text → Preprocessing → Word2Vec Embeddings → BiLSTM Encoder → Hidden States → Classifier
```

### Pipeline B: Text → DistilBERT Encoder → Classifier

```
Input Text → Preprocessing → Tokenization → DistilBERT Encoder → Hidden States → Classifier
```

## Encoder Components

### BiLSTM Encoder

The BiLSTM Encoder processes sequential word embeddings and captures bidirectional context.

#### Architecture Details

- **Input**: Word embeddings from Word2Vec (dimension: 300)
- **LSTM Layers**: 2 stacked bidirectional LSTM layers
- **Hidden Dimension**: 256 per direction (256 total)
- **Dropout**: 0.5 between layers and at output
- **Output**: Concatenated final hidden states from both directions

#### Key Features

- **Bidirectional Processing**: Captures context from both past and future words
- **Sequence Packing**: Efficiently handles variable-length sequences by ignoring padding
- **Layer Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting on training data

#### Input/Output Specification

**Input:**
- `embedded_sequences`: Tensor of shape `[batch_size, seq_len, embedding_dim]`
- `lengths`: Tensor of shape `[batch_size]` containing actual sequence lengths

**Output:**
- `final_hidden`: Tensor of shape `[batch_size, hidden_dim * 2]`

### DistilBERT Encoder

The DistilBERT Encoder uses a pretrained transformer model to generate contextual embeddings.

#### Architecture Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Parameters**: 66 million
- **Hidden Size**: 768
- **Layers**: 6 transformer layers
- **Attention Heads**: 12
- **Max Sequence Length**: 512 tokens

#### Key Features

- **Pretrained Knowledge**: Leverages pretraining on large text corpora
- **Contextual Embeddings**: Each token representation depends on entire sequence
- **Fine-tuning Capability**: Can adapt to domain-specific patterns
- **CLS Token Pooling**: Uses special [CLS] token as sequence representation
- **Layer Normalization**: Applied to output for stability
- **Dropout Regularization**: 0.3 dropout rate at output

#### Input/Output Specification

**Input:**
- `input_ids`: Tensor of shape `[batch_size, seq_len]` containing token IDs
- `attention_mask`: Tensor of shape `[batch_size, seq_len]` (1 for real tokens, 0 for padding)

**Output:**
- `pooled_output`: Tensor of shape `[batch_size, 768]`

## Configuration

### BiLSTM Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `embedding_dim` | 100 | 50-300 | Dimension of input word embeddings |
| `hidden_dim` | 128 | 64-512 | Hidden state dimension per direction |
| `num_layers` | 2 | 1-4 | Number of stacked LSTM layers |
| `dropout` | 0.3 | 0.0-0.5 | Dropout probability |
| `bidirectional` | True | True/False | Use bidirectional processing |

### DistilBERT Hyperparameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `model_name` | distilbert-base-uncased | See HuggingFace | Pretrained model identifier |
| `freeze_bert` | False | True/False | Freeze BERT parameters |
| `dropout` | 0.3 | 0.0-0.5 | Output dropout probability |
| `pooling_strategy` | cls | cls/mean/max | Token pooling method |

### Recommended Configurations

#### For Limited Resources (CPU/Low Memory)

```python
# BiLSTM
encoder = BiLSTMEncoder(
    embedding_dim=50,
    hidden_dim=64,
    num_layers=1,
    dropout=0.2
)

# DistilBERT
encoder = DistilBERTEncoder(
    model_name='distilbert-base-uncased',
    freeze_bert=True,  # Freeze to reduce memory
    dropout=0.2
)
```

#### For High Performance (GPU)

```python
# BiLSTM
encoder = BiLSTMEncoder(
    embedding_dim=300,
    hidden_dim=256,
    num_layers=3,
    dropout=0.3
)

# DistilBERT
encoder = DistilBERTEncoder(
    model_name='distilbert-base-uncased',
    freeze_bert=False,  # Fine-tune for better accuracy
    dropout=0.3
)
```

## Performance Considerations

### Memory Requirements

#### BiLSTM Encoder

- **Model Size**: ~5-10 MB (depending on configuration)
- **Peak Memory**: ~500 MB - 2 GB during training
- **Inference**: ~100-500 MB per batch

#### DistilBERT Encoder

- **Model Size**: ~260 MB
- **Peak Memory**: ~2-8 GB during training
- **Inference**: ~1-2 GB per batch

### Computational Complexity

#### BiLSTM Encoder

- **Time Complexity**: O(n * d^2) where n = sequence length, d = hidden dimension
- **Training Speed**: ~100-500 sequences/second (CPU), ~1000-2000 sequences/second (GPU)
- **Inference Speed**: ~500-1000 sequences/second (CPU), ~2000-5000 sequences/second (GPU)

#### DistilBERT Encoder

- **Time Complexity**: O(n^2 * d) where n = sequence length, d = hidden dimension
- **Training Speed**: ~10-50 sequences/second (CPU), ~100-200 sequences/second (GPU)
- **Inference Speed**: ~50-100 sequences/second (CPU), ~200-500 sequences/second (GPU)

---

Last Updated: October 2025