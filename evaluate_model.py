# evaluate_model.py
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

# 1 Load CSV
df = pd.read_csv("reply_classification_dataset.csv")  

# 2 Normalize labels 
df['label'] = df['label'].str.lower()

# 3 Mapping labels to integers
label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
id_to_label = {v: k for k, v in label_to_id.items()}

df['label_id'] = df['label'].map(label_to_id)

# 4 Loading tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
model.eval()  

# 5 Encode all text
texts = df['reply'].tolist()
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 6 Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    predicted_ids = torch.argmax(outputs.logits, dim=1).tolist()

# 7 Map predicted IDs back to labels
pred_labels = [id_to_label.get(pid, "unknown") for pid in predicted_ids]

# 8 True labels
true_labels = df['label'].tolist()

# 9 Print evaluation metrics
print("Accuracy:", accuracy_score(true_labels, pred_labels))
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, digits=4))



