# Machine Learning CLI Application
**Author:** Chao Li

## Overview

This project implements a complete end-to-end Machine Learning workflow for both regression and classification tasks using Python and scikit-learn.

It demonstrates:

- Data preprocessing
- Model training and evaluation
- Pipeline construction
- Model persistence using pickle
- Interactive CLI-based user input

The project is designed with modular structure and reproducible ML pipelines, simulating real-world ML application development.

This project was developed as part of academic coursework and further refined to demonstrate practical ML engineering skills.

## Key Features

- End-to-end ML pipeline for regression and classification
- Modular project structure (separated training & CLI logic)
- Scikit-learn Pipeline integration
- Model persistence using pickle
- Evaluation metrics for both regression and classification
- Interactive command-line interface

```markdown
## Project Structure

ml_cli_project/
│
├── src/
│   ├── cli_main.py
│   ├── regression.py
│   └── classification.py
│
├── models/
│   └── (saved trained models)
│
├── phishing_dataset.csv
├── requirements.txt
└── README.md
```

## How to Run

- Python version: Python 3.8+
- Install required libraries: `pip install -r requirements.txt`.
- Run the program: `python src/cli_main.py`
- Follow the on-screen menu prompts to select Regression or Classification, and choose to train a model or make predictions.

## Menu Flow Summary

1. Task Selection
1) Regression (estimate a number)
2) Classification (phishing detection)

2. Option Selection
1) Train model
2) Use model for prediction

3. Classification Model Choice(if selected)
1) K-Nearest Neighbors (KNN)
2) Support Vector Machine (SVM)
3) Decision Tree

## Regression Task - California Housing

### Feature Representation
Each sample contains numerical features:
  - Median income
  - House age
  - Average number of rooms
  - Population
  - Latitude and longitude
The target variable is the median house value.

### Dataset
- California Housing dataset provided by scikit-learn
- Automatically downloaded using scikit-learn’s built-in dataset loader: `fetch_california_housing()`

### Model Implemented
- Linear Regression

### Hyperparameters
- None. The model trains directly on the dataset.

### Evaluation
- Metrics displayed:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- The trained pipeline (scaler + model) is saved using pickle.

## Classification Task - Phishing Website Detection

### Feature Representation
Each sample represents a website, described by multiple numerical features extracted from URLs and webpage properties. The target variable indicates whether the website is:
  - Phishing
  - Legitimate

### Dataset
- Phishing Website Detection dataset from Kaggle
- Download steps:
  1. Log in to Kaggle
  2. Search for a phishing website detection dataset on Kaggle (e.g., "Phishing Website Detection Dataset")
  3. Download and extract the CSV file
  4. Place the CSV file in the project directory

### Model Implemented
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree

### Hyperparameters
- KNN: k (number of neighbors)
- SVM：C (regularization parameter)
- Decision Tree: max_depth (maximum depth)

### Evaluation
- Metrics displayed:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- The trained pipeline (scaler + model if applicable) is saved using pickle.

## Future Improvements

- Convert CLI application into a REST API using FastAPI
- Add Docker containerization for reproducible environments
- Integrate a frontend interface
- Add cross-validation and hyperparameter tuning