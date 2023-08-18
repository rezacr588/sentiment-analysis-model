# src/utils.py

import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model(model, path):
    """Save the trained model to a file."""
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved to {path}")

def load_model(path):
    """Load a trained model from a file."""
    if os.path.exists(path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {path}")
        return model
    else:
        logging.warning(f"No model found at {path}")
        return None
