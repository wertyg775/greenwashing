# greenwashing_bert_trainer.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'


import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Check device (should use MPS on M1)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (slower)")

# 1. Load data
print("\nLoading data...")
df = pd.read_csv('data/training_data.csv')
df = df.rename(columns={'combined_text' : 'text'})
print(f"Total: {len(df)} examples")
print(df['label'].value_counts())

# 2. Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

label_map = {'SPECIFIC': 1, 'VAGUE': 0}
train_labels_num = [label_map[l] for l in train_labels]
val_labels_num = [label_map[l] for l in val_labels]

print(f"Training: {len(train_texts)}, Validation: {len(val_texts)}")

# 3. Tokenize
print("\nTokenizing...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 4. Create datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = Dataset(train_encodings, train_labels_num)
val_dataset = Dataset(val_encodings, val_labels_num)

# 5. Load model
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 6. Training args (optimized for M1)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Good for M1
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_mps_device=True,  # Enable M1 GPU
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 7. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n" + "="*60)
print("🚀 TRAINING STARTED")
print("="*60)
print("Expected time on M1: ~10-15 minutes")
print("="*60 + "\n")

trainer.train()

# 8. Evaluate
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)
eval_results = trainer.evaluate()
print(f"\nValidation Accuracy: {eval_results['eval_accuracy']:.2%}")
print(f"Validation F1 Score: {eval_results['eval_f1']:.2%}")

# 9. Save
os.makedirs('./models/specificity', exist_ok=True)



model.save_pretrained('./models/specificity')
tokenizer.save_pretrained('./models/specificity')
print("\n✓ Model saved to './models/specificity'")

