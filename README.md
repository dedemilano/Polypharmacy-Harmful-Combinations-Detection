# Polypharmacy Harmful Combinations Detection

This project aims to predict the risk of hospitalization due to harmful drug combinations (polypharmacy) using machine learning techniques.

## Project Overview

Polypharmacy refers to the use of multiple medications by a patient, which can lead to harmful drug interactions and adverse effects. This project uses a neural network to predict the risk of hospitalization due to these harmful combinations.

## Features

- **Data Loading and Preprocessing**: Loads patient medication data, removes non-polypharmacy records, and prepares features and target variables.
- **Neural Network Model**: A deep learning model built with PyTorch to predict hospitalization risk.
- **Cross-Validation**: Implements K-Fold cross-validation to evaluate model performance.
- **MLflow Integration**: Tracks experiments and logs metrics with MLflow.

## Code Structure

- `main.py`: Main script containing data loading, model training, and evaluation.
  - **Data Loading**: Reads and preprocesses the dataset.
  - **Model Definition**: Defines the neural network architecture.
  - **Training Loop**: Implements the training loop with loss calculation and backpropagation.
  - **Evaluation**: Evaluates the model using cross-validation.
  - **Logging**: Logs experiment metrics using MLflow.
- `PolypharmacyData.csv`: Example dataset (not included in the repository).

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/dedemilano/Polypharmacy-Harmful-Combinations-Detection.git
   cd Polypharmacy-Harmful-Combinations-Detection
