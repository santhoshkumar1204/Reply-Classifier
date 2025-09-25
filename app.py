import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pickle
import os

# 1 Load dataset
df = pd.read_csv("reply_classification_dataset.csv")  
df.dropna(inplace=True)  
texts = df['reply'].astype(str).tolist()
labels = df['label'].astype(str).str.lower().tolist()

# 2 Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# 3 Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# 4 Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 5 Torch Dataset class
class ReplyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReplyDataset(train_encodings, train_labels)
val_dataset = ReplyDataset(val_encodings, val_labels)

# 6 Load model
num_labels = len(le.classes_)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

# 7 Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    eval_strategy="steps",        
    save_strategy="steps",        
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,           
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss"
)

# 8 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 9 Train model
trainer.train()

# 10 Save model, tokenizer, and label encoder
os.makedirs("./trained_model", exist_ok=True)
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

with open("./trained_model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Training complete! Model saved in ./trained_model")


