## Data preprocessing unit 

This file will store all the preprocessing details for our corpus. 

Suggested data preprocessings 
| Step | Description | Rationale for preprocessing | Notes |
|------|-------------|--------------|-------|
| Remove HTML tags | Strip `<html>`, `<a>`, etc. | Emails often contain HTML; noise for model | Use regex or libraries like `BeautifulSoup` |
| Remove URLs | Replace with `<URL>` | Links are frequent in spam | Sometimes keeping as `<URL>` is useful |
| Remove email addresses | Replace with `<EMAIL>` | Stops user-specific leakage , somewhat is noisy data | |
| Remove numbers | Replace with `<NUM>` | Many random numbers don’t add semantic value | Might keep if spam often uses phone numbers |
| Tokenization | subwords | to match DistilBERTs subword level embedding | feed into wordpiece  |
| Lemmatization/Stemming | Reduce words to base form (`running` → `run`) | Normalizes vocabulary | Lemmatization > stemming for readability |
| Whitelisting | Keep only tokens in an **approved vocabulary list**:<br>• currency<br>• file extensions<br>• spammy tokens<br>• email/URL markers<br>• dates<br>• numbers<br>• emoji<br><br>**Base set**: `a–z, A–Z, 0–9, whitespace, . , ! ? ' " : ; - _ ( ) @ # $ % ^ & *` | serve as a base filtering at character level,easier to keep what is wanted rather than exclude a specific list | to add more stuff to whitelist,may drop important rare tokens |
| Handle repeated characters | Normalize (`loooove` → `love`) | Prevents vocab explosion | |
| pruning words below a minimum frequency| pruning < 10   | removes rare words to reduce noise | to adjust prune level based on embedding performance
| file extension | map .pdf .txt to `<FILE>` | some spam email may have download links disguised as file downloads



proposed preprocessing workflow from raw to input for word2vec model
raw email input -> masking of special tokens -> whitelist filtering at character level -> handling of special case ( de-obfuscate , cap character repeat, min word count etc ) -> output cleaned SENTENCES -> feed into word2vec model e