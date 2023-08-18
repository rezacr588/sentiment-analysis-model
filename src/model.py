# src/model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .data_preprocessing import preprocess_text
from .data_loader import load_imdb_data
import os

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()

    def train(self, train_data, train_labels, test_data,test_labels):
        train_data = [preprocess_text(review) for review in train_data]
        test_data = [preprocess_text(review) for review in test_data]

        # Vectorize the data
        X_train = self.vectorizer.fit_transform(train_data).toarray()
        X_test = self.vectorizer.transform(test_data).toarray()

        # Train the model
        self.model.fit(X_train, train_labels)

        # Evaluate the model
        accuracy = self.model.score(X_test, test_labels)
        return accuracy

    def predict(self, review):
        review = preprocess_text(review)
        features = self.vectorizer.transform([review]).toarray()
        prediction = self.model.predict(features)
        return prediction[0]
