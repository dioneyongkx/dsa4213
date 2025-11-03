import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import os


class DistilBERTReloadEval:
    def __init__(self, ckpt_dir="best_ckpts_distilbert", batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size

        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            os.path.join(ckpt_dir, "tokenizer_config.json")
        )

        # Load best finetuned encoder checkpoint
        encoder_path = os.path.join(ckpt_dir, "best_finetune.pt")
        self.encoder = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.encoder.to(self.device)
        self.encoder.eval()

    def load_test_data(self, texts, labels):
        """Tokenize and create a DataLoader for test data"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        dataset = torch.utils.data.TensorDataset(
            enc["input_ids"], 
            enc["attention_mask"], 
            torch.tensor(labels)
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def extract_embeddings(self, loader):
        """Forward pass through DistilBERT to get CLS token embeddings"""
        embeddings = []
        labels = []

        with torch.no_grad():
            for input_ids, attention_mask, y in tqdm(loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.encoder.distilbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                cls_embed = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
                embeddings.append(cls_embed.cpu())
                labels.append(y)

        return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

    def load_logreg(self):
        """Load Logistic Regression head"""
        with open(os.path.join(self.ckpt_dir, "logistic_regression.pkl"), "rb") as f:
            self.logreg = pickle.load(f)
        print("Loaded Logistic Regression classifier.")

    def load_hgb(self):
        """Load HistGradientBoosting head"""
        with open(os.path.join(self.ckpt_dir, "hgb_model.pkl"), "rb") as f:
            self.hgb = pickle.load(f)
        print("Loaded HGB classifier.")

    def eval_model(self, clf, X, y):
        preds = clf.predict(X)
        print("Accuracy:", accuracy_score(y, preds))
        print(classification_report(y, preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y, preds))


# Example usage template
if __name__ == "__main__":
    # Load raw test data (you plug in your dataset!)
    texts = [...]   # <- list of test sentences
    labels = [...]  # <- same length list of ints

    evaluator = DistilBERTReloadEval()

    # Build test dataset
    test_loader = evaluator.load_test_data(texts, labels)

    # Extract embeddings
    X_test, y_test = evaluator.extract_embeddings(test_loader)

    # Evaluate Logistic Regression
    evaluator.load_logreg()
    evaluator.eval_model(evaluator.logreg, X_test, y_test)

    # Evaluate HGB
    evaluator.load_hgb()
    evaluator.eval_model(evaluator.hgb, X_test, y_test)
