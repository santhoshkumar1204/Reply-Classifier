# 🤖 Reply Classification API

This project implements a reply classification pipeline using machine learning and NLP. It classifies replies into categories such as `positive`, `negative`, or `neutral`, and exposes a REST API for predictions using FastAPI.

---

## 🧩 Project Components

- **Baseline Model**: Trained a simple ML model (e.g., Logistic Regression) as a baseline.
- **Transformer Model**: Fine-tuned `DistilBERT` for reply classification.
- **API Deployment**: Wrapped the trained model in a FastAPI service with a `/predict` endpoint.

---

## 📂 Project Structure

ReplyClassifier/
│
├── trained_model/ # Fine-tuned DistilBERT model, tokenizer, and label encoder
├── deploy_app.py # FastAPI deployment code
├── train_model.py # Transformer training code
├── baseline_model.py # Baseline ML model code
├── reply_classification_dataset.csv # Dataset of replies
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy code

---

## 🛠 Setup Instructions (Local)

### 1. Open terminal / command prompt  
### 2. Navigate to the project folder:

```bash
cd path/to/ReplyClassifier
3. Create and activate a virtual environment:
On Windows:
bash
Copy code
python -m venv replyenv
replyenv\Scripts\activate
On Linux / macOS:
bash
Copy code
python3 -m venv replyenv
source replyenv/bin/activate
4. Install dependencies:
bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
🚀 Run the API
Start the FastAPI server using:

bash
Copy code
uvicorn deploy_app:app --reload
Once running, the API will be available at:

cpp
Copy code
http://127.0.0.1:8000
You can visit http://127.0.0.1:8000/docs for the interactive Swagger UI.

🔍 Example Usage
➤ Example Request
POST /predict

Headers:

pgsql
Copy code
Content-Type: application/json
Body:

json
Copy code
{
  "text": "Looking forward to the demo!"
}
➤ Example Response
json
Copy code
{
  "label": "positive",
  "confidence": 0.87
}
📄 Notes
The trained_model folder contains the fine-tuned transformer, tokenizer, and label encoder.

The API returns both the predicted label and the confidence score.

The baseline model and evaluation scripts are included for reference and comparison.

✅ Evaluation
Models are evaluated using accuracy and F1-score.

The baseline model provides a basic benchmark.

The fine-tuned DistilBERT model achieves significantly better performance.

## 🎥 Video Demonstration

👉 Watch a quick [demo of the project](https://drive.google.com/file/d/11eZ4GoAHXmO-moVjdapEj7jJ4A1lCIlO/view?usp=sharing)
