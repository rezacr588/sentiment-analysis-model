# tests/test_model.py
from src.model import SentimentModel
from src.utils import load_model
import logging

def test_sentiment(review, model):
    prediction = model.predict(review)

    if prediction == 1:
        return "Positive Sentiment"
    else:
        return "Negative Sentiment"
    
MODEL_PATH = 'model.pkl'

def main():
    # Check if a pre-trained model exists
    model = load_model(MODEL_PATH)
    if model:
        logging.info("Using the existing model.")
    else:
        logging.info("Training a new model...")
        model = SentimentModel()

    # Get a review from the user
    review = input("Enter a movie review to analyze sentiment: ")

    # Predict and display the sentiment
    sentiment = test_sentiment(review, model)
    print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
