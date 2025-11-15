# Model Evaluation and Cross-Domain Testing

## Directory Overview
This directory contains the evaluation and cross-domain testing pipelines used to measure the performance and generalization ability of our BiLSTM and DistilBERT email scam detection models.

The goal is to test how well models trained on email data transfer to unseen distributions, including other email sources and SMS scam datasets.

Below are the two main evaluation notebooks:

| File | Description |
|----------------|-------------|
| `biLSTM_evaluation_cdtesting.ipynb` | Contains all evaluation and cross-domain testing code for BiLSTM base and ablation models |
| `distilBERT_evaluation_cdtesting.ipynb` | Contains all evaluation and cross-domain testing code for DistilBERT base and ablation models |


## Model Performance Evaluation

Each notebook includes:

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
- Inspecting misclassifcation cases (False Positives and False Negatives)

These results allow a direct comparison between baseline and ablation model variants.


## Cross-Domain Testing

Both notebooks evaluate **generalization to unseen SMS Scam dataset**. The cross-domain pipeline includes:

- Cleaning and tokenizing external datasets
- Running inference through the same trained models
- Reporting all cross-domain metrics (similar to model performance evaluation)
- Visualizing confusion matrices
- Inspecting misclassified examples to understand domain shift failure points

This helps assess model robustness outside the original training distribution.

