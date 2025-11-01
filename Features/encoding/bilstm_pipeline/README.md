# 🧠 BiLSTM Model Pipeline

This subfolder contains the complete training and evaluation pipelines for the **BiLSTM-based spam classification model**.  
It includes modularized code, preprocessing steps, training notebooks, and documentation to ensure reproducibility and clarity.

Due to GitHub storage restrictions, **saved model checkpoints** and **processed datasets** are not included in this repository.  
Throughout the notebooks, these external references are denoted as:
- `best_ckpts/` → directory containing trained BiLSTM model checkpoints  
- `embedder_files/` → directory containing trained **Word2Vec embeddings** and **SentencePiece processor** files  

---

## 📘 Notebook Overview

| File | Description |
|------|--------------|
| **`biLSTM_prepro.ipynb`** | Includes text preprocessing, `Word2Vec` embedding training, and `SentencePiece` subword processing. These are separated from the main pipeline so that `biLSTM_pipeline.ipynb`, `biLSTM_ablation.ipynb`, and `biLSTM_reloader.ipynb` can directly reuse the generated embedding and tokenizer models. Full preprocessing details are documented below and `Word2Vec` and `SentencePiece` are saved to `embedder_files/` |
| **`biLSTM_pipeline.ipynb`** | Contains the end-to-end training workflow for the BiLSTM base model, focusing solely on supervised training of the encoder and classifier head using precomputed embeddings. Produces and saves the final checkpoint to `best_ckpts/`. |
| **`biLSTM_ablation.ipynb`** | Ablation study training pipeline combining the BiLSTM encoder with a `HistGradientBoosting` classifier head. |
| **`biLSTM_reloader.ipynb`** | Reloads both trained models for downstream evaluation and cross-domain testing. |
| **`biLSTM_training_eda.ipynb`** | Used to analyse training results and generate key visualizations referenced in the report. |
| **`biLSTM.py`** | Central BiLSTM module definition. Modularization was intentional to support the “1-notebook-per-pipeline” project design. |

---

## ⚙️ Preprocessing Unit — `biLSTM_prepro.ipynb`

This notebook contains the preprocessing workflow applied to all corpora used in the BiLSTM pipeline.  
The goal was to produce a **clean, standardized text corpus** compatible with both Word2Vec embedding training and sequence modeling.

### Data Preprocessing Details

| Step | Description | Rationale | Notes |
|------|-------------|------------|-------|
| **Remove HTML tags** | Strips tags like `<html>` and `<a>` while preserving semantic info. | Emails often contain HTML noise. | Used BeautifulSoup (`bs4`); `<a>` tags masked via custom function to retain attributes. |
| **Mask URLs** | Replaces all links with `<URL>`. | Links are frequent spam indicators. | Applied both during HTML parsing and separately for non-HTML links. |
| **Mask numbers** | Replaces digits with `<NUMBER>`. | Prevents unnecessary embeddings for unique numeric tokens. | Could alternatively be dropped if deemed uninformative. |
| **Mask money values** | Replaces amounts with `<MONEY>`. | Preserves potential scam cues. | Mapped early to avoid conflicts with other masking steps. |
| **Mask email addresses** | Replaces addresses with `<EMAIL>`. | Prevents user-specific leakage and reduces noise. | Handles multiple domain/user formats; applied during and after HTML parsing. |
| **Lemmatization** | Converts inflected words to their base form. | Normalizes vocabulary, improving generalization. | Preferred over stemming for readability. |
| **Subword tokenization** | Splits text into subword units via SentencePiece. | Ensures alignment with DistilBERT’s subword representation. | Uses trained SentencePiece model (`email_sp.model`). |
| **Character-level whitelisting** | Retains only characters from an approved set (letters, digits, punctuation, emoji). | Provides consistent input space and avoids noisy tokens. | Easier to maintain than blacklisting rare symbols. |
| **Normalize repeated characters** | Reduces excessive character repetition (e.g., `loooove` → `love`). | Prevents vocabulary explosion. | Helps reduce out-of-vocabulary noise. |
| **Prune rare words** | Removes tokens below minimum frequency (e.g., `< 10`). | Eliminates noisy, low-value words from the corpus. | Threshold tunable based on embedding performance. |
| **File extension masking** | Maps `.pdf`, `.txt`, etc., to `<FILE>`. | Captures common phishing/scam download cues. | Handled both during HTML parsing and in later regex mapping. |

---

✅ **Summary:**  
The preprocessing pipeline standardizes multiple text corpora into a consistent, noise-reduced, and semantically meaningful form suitable for **embedding training** and **sequence modeling**.  
This ensures the BiLSTM encoder learns from clean, representative linguistic patterns while preserving key scam signals (URLs, money, files, etc.).