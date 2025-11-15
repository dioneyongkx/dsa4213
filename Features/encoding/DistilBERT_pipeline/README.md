# 🧠 DistilBERT Pipeline

This subfolder contains the complete training, evaluation, and ablation pipelines for the DistilBERT-based spam classification model.
It includes modularized code, preprocessing steps, classifier heads, training notebooks, and documentation to ensure reproducibility and clarity.

Due to GitHub storage and licensing constraints, DistilBERT checkpoints and processed datasets are **not** included in this repository.
Throughout the notebooks, these external references are denoted as:

* `best_ckpts_distilbert/` → directory containing trained DistilBERT model checkpoints
* `datasets/encoder_dataset/clean/distilbert` → directory containing the vocabulary files used for fine-tuning

---

## 📘 Notebook Overview

| File                              | Description                                                                                                                                                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **distilbert_pipeline.ipynb**     | Implements the end-to-end fine-tuning pipeline for DistilBERT using a standard classification head. Produces the final trained checkpoint and evaluation metrics, saved to `best_ckpts/`.                                           |
| **distilbert_ablation.ipynb**     | Runs the ablation pipeline, where DistilBERT embeddings are frozen and fed into alternative classifier heads such as Logistic Regression and HistGradientBoosting.                                                                  |
| **distilbert_reload_eval.ipynb**     | Reloads both base and ablation-trained models for downstream evaluation, analysis, and cross-domain testing.                                                                                                                        |
---

## ⚙️ Preprocessing Unit — distilbert_prepro.ipynb

This notebook contains the preprocessing workflow used to prepare all datasets for the DistilBERT fine-tuning pipeline.
The objective was to maintain semantic richness while ensuring compatibility with transformer-based tokenization.

### **Data Preprocessing Details**

| Step                                       | Description                                                           | Rationale                                                | Notes                                                      |
| ------------------------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- |
| **HTML tag removal**                       | Strip HTML structures while preserving readable text.                 | Emails frequently contain embedded HTML noise.           | Uses BeautifulSoup similar to the BiLSTM pipeline.         |
| **URL masking**                            | Replace all URLs with the `<URL>` token.                              | URLs are high-value spam indicators.                     | Masking before tokenization preserves consistency.         |
| **Email address masking**                  | Replace emails with `<EMAIL>`.                                        | Prevents leakage and simplifies token space.             | Uses a robust regex to capture common email formats.       |
| **Money value masking**                    | Replace currency and numeric money expressions with `<MONEY>`.        | Retains financial-fraud cues without inflating vocab.    | Applied before numeric masking to avoid conflict.          |
| **Number masking**                         | Replace all standalone digits with `<NUMBER>`.                        | Transformers may overfit to specific numeric tokens.     | Disabled for tokens inside `<MONEY>` spans.                |
| **Lowercasing**                            | Converts all text to lowercase.                                       | DistilBERT-base-uncased expects lowercase input.         | Ensures alignment with tokenizer vocabulary.               |
| **Punctuation normalization**              | Standardizes punctuation spacing and removes invisible characters.    | Avoids tokenization inconsistencies.                     | Especially helpful for multilingual email content.         |
| **Character filtering**                    | Retains a fixed whitelist (letters, digits, punctuation, emoji).      | Ensures robustness against malformed Unicode characters. | Parallels the BiLSTM pipeline.                             |
| **Tokenization with DistilBERT tokenizer** | Converts cleaned text into model-ready token IDs and attention masks. | Required input format for transformer fine-tuning.       | Saved outputs reused by all DistilBERT notebooks.          |
| **Sequence truncation/padding**            | Standardizes sequences to a maximum length.                           | Prevents OOM errors and enables efficient batching.      | Max sequence length selected based on corpus distribution. |

---

## ✅ Summary

The DistilBERT preprocessing pipeline provides a consistent, transformer-friendly representation of all email corpora while preserving semantic cues essential for scam detection (URLs, money, file types, sender details).
By aligning masking conventions with the BiLSTM pipeline and enforcing strict text normalization, this workflow ensures that DistilBERT fine-tunes effectively on clean, standardized inputs.

This design supports:

* reliable cross-domain evaluation
* reproducible embedding extraction
* robust ablation with alternative classifier heads
