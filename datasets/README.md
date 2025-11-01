# 📂 Datasets Directory

This folder stores all datasets used throughout the project.  
Each sub-dataset serves a distinct stage in the model pipelines.

---

## 📁 raw_dataset
- Contains the **original dataset** pulled directly from Kaggle.  
- Acts as the source for all downstream processing.  
- No transformations or filtering are applied here.

---

## 📁 word2vec_dataset
- Used exclusively for **Word2Vec embedding training**.  
- `raw/` — contains filtered columns extracted from `raw_dataset` relevant to embedding training.  
- `clean/` — contains preprocessed text (HTML stripped, tokenized, lower-cased, etc.) used for model training.

---

## 📁 encoder_dataset
- Shared dataset for **BiLSTM** and **DistilBERT** model pipelines.  
- `raw/` — minimally filtered version with selected columns from the original dataset.  
- `clean/` — fully preprocessed datasets specific to each model:
  - `bilstm/` — preprocessed for subword tokenization and sequence padding.
  - `distilbert/` — preprocessed for BERT tokenization.

---

## 📁 cross_domain_dataset
- Used for **cross-domain generalization testing**, also shared by both model pipelines  
- `raw/` — minimally filtered version with selected columns from the original dataset.  
- `clean/` — fully preprocessed datasets specific to each model:
  - `bilstm/` — preprocessed for subword tokenization and sequence padding.
  - `distilbert/` — preprocessed for BERT tokenization.

---

🗒️ **Note:**  
All `raw/` folders contain only the minimally processed or filtered text needed for downstream cleaning,  
while `clean/` folders contain fully preprocessed, tokenized, and ready-to-train datasets.