# 🧩 Dataset Filtering and Raw Data handling notebook

This notebook prepares the **base raw datasets** used for all downstream preprocessing and model training.  
Each dataset originates from a different source and is filtered to retain only the relevant text and label columns, reducing load times
The filtered outputs are stored in their respective `/raw/` directories for further cleaning.

---

## 📁 Dataset Processing Summary

### 📨 Enron Fraud Dataset (`enron_data_fraud_labeled.csv`)
- **Purpose:** Used for **Word2Vec embedding training**.  
- **Filtering:** Selects only the columns `['Body', 'Label']`.  
- **Output path:** `datasets/word2vec_dataset/raw/`  
- Ensures a clean and lightweight corpus for unsupervised embedding training.

---

### 🎯 Phishing Email Dataset (`phishing_email.csv`)
- **Purpose:** Shared **encoder dataset** for both BiLSTM and DistilBERT pipelines.  
- **Preprocessing:**  
  - Removes duplicate entries.  
  - Performs **train/validation/test split** to ensure consistent splits across both models.  
- **Output path:** `datasets/encoder_dataset/raw/`  
- Designed to be the primary labeled dataset for supervised training.

---

### ✉️ SMS Spam Dataset (`spam.csv`)
- **Purpose:** Used for **cross-domain evaluation**.  
- **Filtering:** Selects only `['v2', 'v1']` — text and label columns.  
- **Output path:** `datasets/cross_domain_dataset/raw/`  
- Serves as an external test domain to evaluate model generalization.

---

✅ Each dataset in `/raw/` represents the **filtered but unprocessed** text —  
ready for the next preprocessing stage within each model pipeline.