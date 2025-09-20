## Data preprocessing unit 

This file will store all the preprocessing details for our corpus. 

Suggested data preprocessings 
1. Tokenisation : subword level 
2. Lemmitization : running to run 
3. minimum frequency cut off : 10 first 
4. Noise filtering (whitelist): 
    - currency
    - file extensions
    - spammy tokens
    - email and url markers
    - dates 
    - numbers
    - emoji? 
    - base = [a-A,0-9,!@#$%^&*()]


proposed preprocessing workflow from raw to input for word2vec model
raw email input -> masking of special tokens -> whitelist filtering at character level -> handling of special case ( de-obfuscate , cap character repeat, min word count etc ) -> output cleaned SENTENCES -> feed into word2vec model e