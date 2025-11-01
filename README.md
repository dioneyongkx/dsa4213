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


## Project Directory Overview

This repository is organized into modular components to support a clean and reproducible machine learning workflow.  
Each main directory corresponds to a major stage of the project pipeline — from raw data handling to model feature extraction and evaluation.

| Folder / File | Description |
|----------------|-------------|
| `datasets/` | Contains all datasets used in this project, including raw, cleaned, and processed subsets for each model pipeline. |
| `Features/` | Houses all source code, including preprocessing, BiLSTM, and DistilBERT pipelines. |
| `requirements.txt` | Lists all dependencies required to reproduce the project environment. |

