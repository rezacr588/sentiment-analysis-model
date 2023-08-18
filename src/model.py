# src/model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from .data_preprocessing import preprocess_text

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()

    def train(self, data_path):
        data = pd.read_csv(data_path)
        data['review'] = data['review'].apply(preprocess_text)
        X = self.vectorizer.fit_transform(data['review']).toarray()
        y = data['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def predict(self, review):
        review = preprocess_text(review)
        features = self.vectorizer.transform([review]).toarray()
        prediction = self.model.predict(features)
        return prediction[0]
