import numpy as np
import pandas as pd
import re
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import Configuration, UniformFloatHyperparameter, ConfigurationSpace
import joblib
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

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

# Load the pre-trained models
svm_model = joblib.load('SVM_model.pkl')
lr_model = joblib.load('Logistic Regression_model.pkl')
rf_model = joblib.load('Random Forest_model.pkl')
xgb_model = joblib.load('XGBoost_model.pkl')

models = {
    "SVM": svm_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Fit the classifiers and save the fitted models
def fit_model(model, X_train_tfidf, y_train):
    model.fit(X_train_tfidf, y_train)
    return model

fitted_models = Parallel(n_jobs=-1)(delayed(fit_model)(model, X_train_tfidf, y_train) for model in models.values())

# Define soft voting function
def soft_voting(weights, X):
    final_prob = np.zeros((X.shape[0], 2))  # Assuming binary classification
    for weight, model in zip(weights, models.values()):
        prob = model.predict_proba(X)
        final_prob += weight * prob
    return final_prob

# Define the training function for SMAC
def train(config: Configuration, seed: int = 42) -> float:
    weights = np.array([config["weight1"], config["weight2"], config["weight3"], config["weight4"]])
    weights /= weights.sum()  # Normalize weights to sum to 1
    final_prediction = soft_voting(weights, X_test_tfidf)
    final_class = np.argmax(final_prediction, axis=1)
    return -accuracy_score(y_test, final_class)  # We negate the accuracy to minimize

# Define configuration space
configspace = ConfigurationSpace()
configspace.add_hyperparameters([
    UniformFloatHyperparameter("weight1", 0, 1),
    UniformFloatHyperparameter("weight2", 0, 1),
    UniformFloatHyperparameter("weight3", 0, 1),
    UniformFloatHyperparameter("weight4", 0, 1)
])

# Define SMAC scenario
scenario = Scenario(
    configspace,
    n_trials=150,
    deterministic=True,
    n_workers=-1  # Adapt based on your machine capabilities
)

smac = HyperparameterOptimizationFacade(
    scenario,
    train,
    overwrite=True
)

incumbent = smac.optimize()

# Get and normalize the optimized weights
optimized_weights = np.array([incumbent["weight1"], incumbent["weight2"], incumbent["weight3"], incumbent["weight4"]])
optimized_weights /= optimized_weights.sum()
print("Optimized Weights:", optimized_weights)

# Function to make a final prediction using optimal weights
def predict_with_optimal_weights(text):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform([text])
    prob = soft_voting(optimized_weights, text_tfidf)
    final_prediction = np.argmax(prob, axis=1)
    return 'Fake' if final_prediction[0] == 0 else 'Real'

# Example usage
news_article = "Your news article text here"
print(predict_with_optimal_weights(news_article))

# Evaluate on test set
final_prob_test = soft_voting(optimized_weights, X_test_tfidf)
final_pred_test = np.argmax(final_prob_test, axis=1)
accuracy = accuracy_score(y_test, final_pred_test)
precision = precision_score(y_test, final_pred_test, average='binary')
recall = recall_score(y_test, final_pred_test, average='binary')
f1 = f1_score(y_test, final_pred_test, average='binary')
mcc = matthews_corrcoef(y_test, final_pred_test)
print(f'Final Meta Model Accuracy: {accuracy:.5f}')
print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")
print(f"MCC: {mcc:.5f}")

# Save the meta-model and optimized weights
meta_model = {
    'vectorizer': vectorizer,
    'models': models,
    'optimized_weights': optimized_weights
}
joblib.dump(meta_model, 'meta_model.pkl')

# Function to load the meta-model and make predictions
def load_meta_model_and_predict(text):
    meta_model = joblib.load('meta_model.pkl')
    vectorizer = meta_model['vectorizer']
    models = meta_model['models']
    optimized_weights = meta_model['optimized_weights']
    print(optimized_weights)
    
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform([text])
    prob = soft_voting(optimized_weights, text_tfidf)
    final_prediction = np.argmax(prob, axis=1)
    return 'Fake' if final_prediction[0] == 0 else 'Real'

# Example usage after loading meta-model
print(load_meta_model_and_predict(news_article))