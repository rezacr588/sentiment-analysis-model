# src/data_loader.py

import os

def load_imdb_data(data_dir):
    """Load IMDb movie reviews from the aclImdb dataset."""
    data = []
    labels = []

    # Load positive reviews
    pos_dir = os.path.join(data_dir, 'pos')
    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as file:
            data.append(file.read())
            labels.append(1)  # 1 for positive

    # Load negative reviews
    neg_dir = os.path.join(data_dir, 'neg')

    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as file:
            data.append(file.read())
            labels.append(0)  # 0 for negative

    return data, labels
