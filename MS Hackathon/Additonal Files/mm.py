import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
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

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['text'].values)
joblib.dump(vectorizer, 'vectorizer.pkl')
y = df['label'].values

# Load pre-trained models
# models = {
#     "SVM": joblib.load('SVM_model.pkl'),
#     "Logistic Regression": joblib.load('Logistic Regression_model.pkl'),
#     #"Random Forest": joblib.load('Random Forest_model.pkl')
# }

# # Create a VotingClassifier with soft voting
# voting_clf = VotingClassifier(
#     estimators=[(name, model) for name, model in models.items()],
#     voting='soft'
# )

# # No need to fit the VotingClassifier since the base models are already trained
# # Predict the results using the voting classifier
# voting_clf.fit(X_tfidf, y)
# y_pred = voting_clf.predict(X_tfidf)
# accuracy = accuracy_score(y, y_pred)
# print(f"Meta-Model Accuracy: {accuracy}")
# print("Meta-Model Classification Report:")
# print(classification_report(y, y_pred))

# # Save the meta-model
# joblib.dump(voting_clf, 'mm.pkl')
# print("Meta-model saved as 'mm.pkl'")

# # Function to load the meta-model and make a prediction
# def predict_news(text):
#     clf = joblib.load('mm.pkl')
#     text = preprocess_text(text)
#     text_tfidf = vectorizer.transform([text])
#     prediction = clf.predict(text_tfidf)
#     return 'Fake' if prediction[0] == 0 else 'Real'

# # Example usage
# news_article = "Your news article text here"
# print(predict_news(news_article))
