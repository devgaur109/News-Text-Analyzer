# Import necessary libraries
import pandas as pd
import numpy as np
import re
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('train.csv')

# Preprocessing function to clean the text
def preprocess_text(text):
    if isinstance(text, float):  # Check if the text is a float (missing values)
        return ""
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = df['text'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define classifiers
classifiers = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True,kernel='rbf'),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    # Save the model
    joblib.dump(clf, f'{name}_model.pkl')
    print(f"{name} model saved as '{name}_model_rbf_kernel.pkl'")

# Function to load a model and make a prediction
def predict_news(text, model_name):
    clf = joblib.load(f'{model_name}_model.pkl')
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)
    return 'Fake' if prediction[0] == 0 else 'Real'

# Example usage
news_article = "Your news article text here"
print(predict_news(news_article, "Logistic Regression"))
