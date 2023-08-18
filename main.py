# main.py

import os
from src.model import SentimentModel
from src.data_loader import load_imdb_data
from src.utils import save_model, load_model
import logging

MODEL_PATH = 'saved_model.pkl'
DATA_PATH = 'data/raw/aclImdb'

def main():
    # Check if a pre-trained model exists
    model = load_model(MODEL_PATH)
    if model:
        logging.info("Using the existing model.")
    else:
        logging.info("Training a new model...")
        model = SentimentModel()

    # Load and preprocess the IMDb dataset
    train_data, train_labels = load_imdb_data(os.path.join(DATA_PATH, 'train'))
    test_data, test_labels = load_imdb_data(os.path.join(DATA_PATH, 'test'))    
    # Train the model
    accuracy = model.train(train_data, train_labels, test_data, test_labels)
    logging.info(f"Training accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
