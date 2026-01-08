# specificity_trainer.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =========================================================
# 1. DEVICE SETUP
# =========================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (slower)")

# =========================================================
# 2. LOAD DATA
# =========================================================
print("\nLoading sentence-level data...")
df = pd.read_csv("data/training_sentences.csv")
df = df.rename(columns={"sentence": "text"})

print(f"Total samples: {len(df)}")
print("Label distribution:")
print(df["label_id"].value_counts())

# =========================================================
# 3. TRAIN / VAL SPLIT (sentence-level)
# =========================================================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# =========================================================
# 4. TOKENIZATION
# =========================================================
print("\nTokenizing text...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=128
)

# =========================================================
# 5. DATASET CLASS
# =========================================================
class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentenceDataset(train_encodings, train_labels)
val_dataset = SentenceDataset(val_encodings, val_labels)

# =========================================================
# 6. LOAD MODEL
# =========================================================
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    id2label={0: "VAGUE", 1: "SPECIFIC"},
    label2id={"VAGUE": 0, "SPECIFIC": 1}
)

# =========================================================
# 7. TRAINING ARGUMENTS
# =========================================================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps=100,

    eval_strategy="epoch",   # must set for load_best_model_at_end
    save_strategy="epoch",         # must match evaluation_strategy
    load_best_model_at_end=True,   # will work now
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    use_mps_device=True,           # Apple M1 GPU
)

# =========================================================
# 8. METRICS
# =========================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }

# =========================================================
# 9. TRAIN
# =========================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n" + "="*60)
print("🚀 TRAINING STARTED")
print("Expected time on M1: ~8–12 minutes")
print("="*60 + "\n")

trainer.train()

# =========================================================
# 10. FINAL EVALUATION
# =========================================================
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

eval_results = trainer.evaluate()
print(f"Accuracy:  {eval_results['eval_accuracy']:.2%}")
print(f"F1 (macro): {eval_results['eval_f1_macro']:.2%}")
print(f"Precision: {eval_results['eval_precision_macro']:.2%}")
print(f"Recall:    {eval_results['eval_recall_macro']:.2%}")

# =========================================================
# 11. SAVE MODEL
# =========================================================
os.makedirs("./models/specificity", exist_ok=True)

model.save_pretrained("./models/specificity")
tokenizer.save_pretrained("./models/specificity")

print("\n✓ Model saved to ./models/specificity")
