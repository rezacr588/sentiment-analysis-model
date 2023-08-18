# tests/test_preprocessing.py

from src.data_preprocessing import preprocess_text

def test_preprocess_text():
    assert preprocess_text("Hello World!") == "hello world"
    # Add more test cases as needed
