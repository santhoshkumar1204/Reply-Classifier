# baseline_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1 Load dataset
df = pd.read_csv("reply_classification_dataset.csv")
df.dropna(inplace=True)

texts = df['reply'].astype(str).tolist()
labels = df['label'].str.lower().tolist()  # normalize labels

# 2 Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
print("Label mapping:", dict(zip(le.classes_, range(len(le.classes_)))))

# 3 Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# 4 TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# 5 Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

# 6 Evaluate
y_pred = clf.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_val, y_pred, target_names=le.classes_))
