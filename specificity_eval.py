# load_and_save_results.py

from transformers import Trainer, BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('./models/specificity')
tokenizer = BertTokenizer.from_pretrained('./models/specificity')

# Recreate val_dataset (same as training)
df = pd.read_csv('data/training_data.csv')
df = df.rename(columns={
    'combined_text' : 'text'
})

_, val_texts, _, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

label_map = {'SPECIFIC': 1, 'VAGUE': 0}
val_labels_num = [label_map[l] for l in val_labels]

val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

val_dataset = Dataset(val_encodings, val_labels_num)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Create trainer
trainer = Trainer(
    model=model, 
    eval_dataset=val_dataset,
    compute_metrics= compute_metrics
    )

# Evaluate
results = trainer.evaluate()

# Save
with open('./models/specificity/evaluation_results.txt', 'w') as f:
    f.write(f"Validation Accuracy: {results['eval_accuracy']:.2%}\n")
    f.write(f"Validation F1 Score: {results['eval_f1']:.2%}\n")
    f.write(f"Validation Precision: {results['eval_precision']:.2%}\n")
    f.write(f"Validation Recall: {results['eval_recall']:.2%}\n")

print("✓ Results saved!")
print(f"Accuracy: {results['eval_accuracy']:.2%}")
print(f"F1: {results['eval_f1']:.2%}")