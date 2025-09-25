# Reply Classification API

This project implements a reply classification pipeline using machine learning and NLP. It classifies replies into categories such as `positive`, `negative`, or `neutral` and exposes a REST API for predictions.

# 🧩 Project Components

- Baseline Model: Trained a simple ML model (e.g., Logistic Regression) as a baseline.
- Transformer Model: Fine-tuned `DistilBERT` for reply classification.
- API Deployment: Wrapped the trained model in a FastAPI service with a `/predict` endpoint.

# 📂 Project Structure

ReplyClassifier/
│
├── trained_model/ # Fine-tuned DistilBERT model, tokenizer, and label encoder
├── deploy_app.py # FastAPI deployment code
├── train_model.py # Transformer training code
├── baseline_model.py # Baseline ML model code
├── reply_classification_dataset.csv # Dataset of replies
├── requirements.txt # Dependencies
└── README.md

## 🛠 Setup Instructions (Local)

1. Open terminal / command prompt  

2. Navigate to the project folder 

cd path/to/ReplyClassifier

#Create a virtual environment

python -m venv replyenv

#Activate the virtual environment

Windows:

replyenv\Scripts\activate

Linux / Mac:

source replyenv/bin/activate

#Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

🚀 Run the API
Start the FastAPI server:

uvicorn deploy_app:app --reload
The API will be available at: http://127.0.0.1:8000

Use the /predict endpoint to classify replies.

Example Request

POST /predict
Content-Type: application/json

{
  "text": "Looking forward to the demo!"
}
Example Response
json

{
  "label": "positive",
  "confidence": 0.87
}

📄 Note:

-The trained_model folder contains the fine-tuned transformer, tokenizer and label encoder.

-The API returns the predicted label and confidence score.

-The baseline model and evaluation scripts are included for reference and comparison.

✅ Evaluation:

Accuracy and F1-score are used to evaluate models.

Fine-tuned DistilBERT provides higher performance than the baseline.

# 🎥Video Demonstration

Here is a Quick demo of the project : [Demonstration](https://drive.google.com/file/d/11eZ4GoAHXmO-moVjdapEj7jJ4A1lCIlO/view?usp=sharing)
