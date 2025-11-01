This subfolder contains the training pipelines for our BiLSTM model. It mainly consists of notebooks and a main predefined BiLSTM module for easy reference. Due to github restrictions, our saved models and clean data cannot be put onto github. In each pipeline, the references to our saved models will be denoted as `best_ckpts` which contains the trained BiLSTM model and `full_220` which contains our trained Word2Vec embeddings and Sentencepiece Processor

1) `biLSTM_prepro.ipynb` : This notebook serves to clean and prepare all the dataset involved in the BiLSTM pipeline, details of the preprocessing are included in full below
1) `biLSTM_pipeline.ipynb` : The main training pipeline for our base BiLSTM model, the final model is saved as a checkpoint 
2) `biLSTM_ablation.ipynb` : The ablation training pipeline for our BiLSTM encoder + HistGradientBoosy classifier head 
3) `biLSTM_reloader.ipynb` : This notebook was designed to reload both models for downstream evaluation testing
4) `biLSTM_training_eda.ipynb` : This notebook was used to analyse the training data and relevant plots included in our report 
5) `biLSTM.py` : This module serves as the central source of our BiLSTM module, the decision to modularise our model definition was intentional so as to support our 1 notebook per pipeline project design





## Data preprocessing unit 

This file will store all the preprocessing details for our corpus. 

Suggested data preprocessings 
| Step | Description | Rationale for preprocessing | Notes |
|------|-------------|--------------|-------|
| Remove HTML tags | Strip `<html>`, `<a>`, etc. | Emails often contain HTML; noise for model | BeautifulSoup from bs4 was used to parse html,`<a>` tags were handled with a masking function to preserve information for various attributes |
| Remove URLs | Replace with `<URL>` | Links are frequent in spam | mapped once in HTML parsing and mapped again to handle non-`<a>` cases in HTML parser|
| Remove numbers | Replace with `<NUMBER>` | prevent mapping one embedding to every number| consider to drop completely as it does not bring any value |
| Remove money | Replace with `<MONEY>` | special token to preserve scam signal| mapped at higher levels to deconflict with deobufuscate |
| Remove email addresses | Replace with `<EMAIL>` | Stops user-specific leakage , somewhat is noisy data | mapped once in HTML parsing and mapped again to handle non-`<a>` cases in HTML parser , currently can handle different end domains and username configuration|
| Tokenization | subwords | to match DistilBERTs subword level embedding | feed into wordpiece  |
| Lemmatization/Stemming | Reduce words to base form (`running` → `run`) | Normalizes vocabulary | Lemmatization > stemming for readability |
| Whitelisting | Keep only tokens in an **approved vocabulary list**:<br>• currency<br>• file extensions<br>• spammy tokens<br>• email/URL markers<br>• dates<br>• numbers<br>• emoji<br><br>**Base set**: `a–z, A–Z, 0–9, whitespace, . , ! ? ' " : ; - _ ( ) @ # $ % ^ &` | serve as a base filtering at character level,easier to keep what is wanted rather than exclude a specific list | to add more stuff to whitelist,may drop important rare tokens |
| Handle repeated characters | Normalize (`loooove` → `love`) | Prevents vocab explosion | |
| pruning words below a minimum frequency| pruning < 10   | removes rare words to reduce noise | to adjust prune level based on embedding performance
| file extension | map .pdf .txt to `<FILE>` | some spam email may have download links disguised as file downloads |mapped once in HTML parsing and mapped again to handle non-`<a>` cases in HTML parser| 


