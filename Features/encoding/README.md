# Email Scam Detection - Encoding Layer Documentation

## Overview

This module implements the encoding layer for email scam detection as part of the DSA4213 Final Project. The encoding layer is responsible for transforming preprocessed email text into fixed-dimensional vector representations that capture semantic and contextual information.

Two distinct encoder architectures are implemented:

- **BiLSTM Encoder**: A bidirectional Long Short-Term Memory network that processes Word2Vec embeddings
- **DistilBERT Encoder**: A pretrained transformer-based encoder using DistilBERT

These encoders serve as the core feature extraction component in the scam detection pipeline, bridging the gap between raw text input and the final classification layer.

## Architecture

### Main Pipeline A: Word2Vec → BiLSTM Encoder → Classifier

```
Input Text → Preprocessing → Word2Vec Embeddings → BiLSTM Encoder → Hidden States → Classifier
```

### Main Pipeline B: Text → DistilBERT Encoder → Classifier

```
Input Text → Preprocessing → Tokenization → DistilBERT Encoder → Hidden States → Classifier
```


---
### Ablation Pipeline A: BiLSTM Encoder → HistGradientBoost Classifier

```
Frozen BiLSTM Encoder → Hidden States → HistGradientBoost Classifier
```

### Ablation Pipeline B: DistilBERT → HistGradientBoost Classifier

```
DistilBERT Encoder → Hidden States → HistGradientBoost Classifier
 ```