# src/data_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and tokenize
    text = text.lower().split()
    
    # Remove stopwords and stem
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    
    # Join tokens and return
    return ' '.join(text)
