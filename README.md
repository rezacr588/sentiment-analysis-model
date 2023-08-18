# Sentiment Analysis Model

This repository contains a machine learning model for sentiment analysis. The model is trained on the IMDb movie reviews dataset and can predict whether a given review has a positive or negative sentiment.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Trained on the IMDb movie reviews dataset.
- Uses a logistic regression algorithm for prediction.
- Provides utilities for data preprocessing and vectorization.
- Includes a testing script to evaluate the model's performance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rezacr588/sentiment-analysis-model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis-model
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. To train the model (if a pre-trained model doesn't exist):
   ```bash
   python main.py
   ```

2. To test the model with your own reviews:
   ```bash
   python tests/test_model.py
   ```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
