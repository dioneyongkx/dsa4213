# DSA4213 Group 36


This is the complete code repository for our DSA4213 final project.  

Authors: **Dione Yong**
          **Javier Goh**
          **Subhashree Panneer**

This project explores the possibility of using NLP algorithms to identify scam messages.

Scam detection in emails has become an increasingly urgent challenge in the digital age. With the rise of online communication, email remains one of the most common vectors for fraudulent activity. We aim to train NLP algorithms to be able to recognise and flag out such scams.


A brief overview of our project:
1. Enron Fraud Email Dataset wil be used to train the word embedding layer of our pipeline
    https://www.kaggle.com/datasets/advaithsrao/enron-fraud-email-dataset/data
2. We will be comparing two different model pipelines, their efficiency and accuracy will be evaluated
3. Finally, we will conduct some ablation studies on our best perfoming model


## Navigating our project
Please follow the steps below to reproduce the results of this project
| **Step** | **Action** | **Details / File Paths** |
|---------|------------|---------------------------|
| **1** | **Download Raw Datasets** | Download the 3 CSV datasets and place them into: `datasets/raw_datasets/` <br> link 1 : [enron_fraud_labeled.csv](https://www.kaggle.com/datasets/advaithsrao/enron-fraud-email-dataset?resource=download&select=enron_data_fraud_labeled.csv)  <br> link 2 : [phishing_email.csv](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data?select=phishing_email.csv) <br> link 3 : [spam.csv](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| **2** | **Install Dependencies** | Run: `pip install -r requirements.txt` |
| **3** | **Prepare Raw Data** | Execute `Features/raw_data_handler/raw_data_handling.ipynb`<br>This will generate cleaned *raw splits* for:<br>• `datasets/word2vec_dataset/raw/`<br>• `datasets/encoder_dataset/raw/`<br>• `datasets/cross_domain/raw/` |
| **4** | **Run BiLSTM Pipeline** | Navigate to: `Features/encoding/bilstm_pipeline/`<br>Run in order:<br>1. `biLSTM_prepro.ipynb` — preprocessing + SentencePiece + Word2Vec<br>2. `biLSTM_pipeline.ipynb` — full BiLSTM training + threshold tuning<br>3. `biLSTM_ablation.ipynb` — HGB head ablation on frozen encoder |
| **5** | **Run DistilBERT Pipeline** | Navigate to: `Features/encoding/DistilBERT_pipeline/`<br>Run in order:  |
| **6** | **Run Evaluation Pipeline** | Navigate to: `Features/evaluation/`<br>Run in order: |

## Project Directory Overview

This repository is organized into modular components to support a clean and reproducible machine learning workflow.  
Each main directory corresponds to a major stage of the project pipeline — from raw data handling to model feature extraction and evaluation.

| Folder / File | Description |
|----------------|-------------|
| `datasets/` | Contains all datasets used in this project, including raw, cleaned, and processed subsets for each model pipeline. |
| `Features/` | Houses all source code, including preprocessing, BiLSTM, and DistilBERT pipelines. |
| `requirements.txt` | Lists all dependencies required to reproduce the project environment. |

---
