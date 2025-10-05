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


ACTUAL workflow from raw input to word embeddding 

1) raw file is truncated (not confirmed) , only taking body and label columns, stored as df
2) df is passed into preprocess_email_text(), the main preprocessing pipeline, the details of the preprocessing pipeline is above , returns a df of same dimensions
3) df is passed into vocab_builder to generate subword level vocab,generates .model and .vocab files
4) df is passed into vocab_to_id_mapper(), adds 2 new columns : sp_id and padded_sp_id (for downstream BiLSTM) , sentence piece id is used because it is fixed , more consistent 
5) df is pass into word2vec_embedder() , function trains, and generates embeddings, as well as a sp_id -> wv vector mapping 


ACTUAL workflow dimensions

1) word2vec 
    * vector_dim = 300 (to match with LSTM input_dim)

2) Sentence piece subword processor 
    * max_len = 256 ( padded if short, truncated if longer)
    * vocab_size = 50_000
    * encoder model = bpe (Byte pair encoding)