# deploy_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pickle

# 1 Load model, tokenizer, and label encoder

try:
    tokenizer = DistilBertTokenizerFast.from_pretrained("./trained_model")
    model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
    model.eval()

    with open("./trained_model/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

except Exception as e:
    print("Error loading model or label encoder:", e)
    raise e


# 2 FastAPI app setup
app = FastAPI(
    title="Reply Classification API",
    description="API to classify prospect replies into positive, negative, or neutral",
    version="1.0.0"
)

class InputText(BaseModel):
    text: str


# 3 Predict endpoint

@app.post("/predict")
def predict(input: InputText):
    """
    Predict the class of a given reply text.
    Input JSON: {"text": "some reply text"}
    Output JSON: {"label": "positive", "confidence": 0.87}
    """
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Tokenize input
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx].item()

    return {"label": pred_label, "confidence": round(confidence, 3)}

# 4 Optional health check endpoint
@app.get("/")
def root():
    return {"message": "Reply Classification API is running. Use /predict endpoint."}
