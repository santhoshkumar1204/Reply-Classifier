from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pickle

# Load tokenizer and trained model
model_path = "./trained_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Load saved label encoder
with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Test input 
text = "ABsolutely ! Lets do this"

# Convert text to model input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Get model predictions
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()

# Convert numeric class back to label
predicted_label = label_encoder.inverse_transform([predicted_class])[0]
print(f"Predicted Label: {predicted_label}")
