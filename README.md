# Polypharmacy Harmful Combinations Detection

This project aims to predict the risk of hospitalization due to harmful drug combinations (polypharmacy) using machine learning techniques.

## Project Overview

Polypharmacy refers to the use of multiple medications by a patient, which can lead to harmful drug interactions and adverse effects. This project uses a neural network to predict the risk of hospitalization due to these harmful combinations.

## Features

- **Data Loading and Preprocessing**: Loads patient medication data, removes non-polypharmacy records, and prepares features and target variables.
- **Neural Network Model**: A deep learning model built with PyTorch to predict hospitalization risk.
- **Cross-Validation**: Implements K-Fold cross-validation to evaluate model performance.
- **MLflow Integration**: Tracks experiments and logs metrics with MLflow.

## Machine Learning Techniques

- **Neural Networks**: Utilizes a deep learning model to predict the risk of hospitalization.
- **Random Forest**: Uses a random forest classifier to evaluate feature importance and model performance.
- **Association Rules**: Applies association rule mining to identify frequent drug combinations and their potential risks.

## Code Structure

- `main.py`: Main script containing data loading, model training, and evaluation.
  - **Data Loading**: Reads and preprocesses the dataset.
  - **Model Definition**: Defines the neural network architecture.
  - **Training Loop**: Implements the training loop with loss calculation and backpropagation.
  - **Evaluation**: Evaluates the model using cross-validation.
  - **Logging**: Logs experiment metrics using MLflow.
- `PolypharmacyData.csv`: Example dataset (not included in the repository).

## Prerequisites

- Python 3.7 or higher
- PyTorch
- MLflow
- Pandas
- Scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/dedemilano/Polypharmacy-Harmful-Combinations-Detection.git
   cd Polypharmacy-Harmful-Combinations-Detection
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## How to Run

1. Ensure you have the dataset `PolypharmacyData.csv` in the project directory.
2. Run the main script:
   ```sh
   python main.py
   ```

## Usage

- Modify the `main.py` script to load your own dataset and adjust model parameters as needed.
- Use MLflow to track and visualize your experiments.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
