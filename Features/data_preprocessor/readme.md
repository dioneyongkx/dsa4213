## Data preprocessing unit 

This file will store all the preprocessing details for our corpus. 

Suggested data preprocessings 
1. Tokenisation : subword level? BERT is subword level if want to match
2. Lemmitization : running to run 
3. minimum frequency cut off 
4. Noise filtering : for this specifically, i think we should go with a ' what to keep approach '  rather than ' what to throw approach '  . For this portion also have to think about the filtering granularity -> filter at character level or filter at word level, word level is essentially harder because have to anticipate all sorts of words

    noise filtering whitelist
    - currency
    - file extensions
    - spammy tokens
    - email and url markers
    - dates 
    - numbers
    - emoji? 


proposed preprocessing workflow from raw to input for word2vec model
raw email input -> masking of special tokens -> whitelist filtering at character level -> handling of special case ( de-obfuscate , cap character repeat, min word count etc ) -> output cleaned SENTENCES -> feed into word2vec model 