# Model Evaluation and Cross-Domain Testing

## Directory Overview
This directory contains the evaluation and cross-domain testing pipelines used to measure the performance and generalization ability of our BiLSTM and DistilBERT email scam detection models.

The goal is to test how well models trained on email data transfer to unseen distributions, including other email sources and SMS scam datasets.

Below are the two main evaluation notebooks (other .png files in the repository can be ignored):

| File | Description |
|----------------|-------------|
| `word2vec_evaluation.ipynb` | Contains all word2vec evaluation for main BiLSTM model |
| `biLSTM_evaluation_cdtesting.ipynb` | Contains all evaluation and cross-domain testing code for BiLSTM base and ablation models |
| `distilBERT_evaluation_cdtesting.ipynb` | Contains all evaluation and cross-domain testing code for DistilBERT base and ablation models |


## Model Performance Evaluation

The **Word2Vec evaluation** notebook includes:

- Loading the trained Word2Vec embedding model
- Loading the encoded datasets for both Batch A and Batch B
- Computing global embedding quality metrics:
  - Cosine similarity between word pairs
  - Word analogy evaluations (where applicable)
- Visualizing embedding structure to inspect semantic relationships using:
  - PCA (Principal Component Analysis)
  - t-SNE dimensionality reduction
  - PCA → t-SNE pipeline for improved clustering

These evaluations help validate the semantic consistency of the Word2Vec embeddings and ensure they capture meaningful relationships prior to being used in downstream classification models.<br><br>



The **BiLSTM and DistilBERT evaluation** notebooks include:

- Loading of model checkpoints (`best_ckpts/` and `best_ckpts_distilbert/`)
- Loading SentencePiece / HuggingFace tokenizers
- Feature extraction from BiLSTM or DistilBERT encoders
- Running inference on the evaluation dataset
- Computing performance evaluation metrics:
  - Accuracy  
  - Precision, Recall  
  - F1-score  
  - ROC-AUC  
  - Confusion Matrix
- Inspecting misclassified examples (False Positives & False Negatives) to better understand weaknesses of the models

These results allow a direct comparison between baseline and ablation model variants.


## Cross-Domain Testing

Both notebooks evaluate **generalization to unseen SMS Scam dataset**. The cross-domain pipeline includes:

- Cleaning and tokenizing external datasets
- Running inference through the same trained models
- Reporting all cross-domain metrics (similar to model performance evaluation)
- Visualizing confusion matrices
- Inspecting misclassified examples to understand domain shift failure points

This helps assess model robustness outside the original training distribution.

