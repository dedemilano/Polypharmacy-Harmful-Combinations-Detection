import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import hydra
from omegaconf import DictConfig

# Configuration
class Config:
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    K_FOLDS = 5

# Set random seeds for reproducibility
np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)

class PolypharmacyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

def load_and_preprocess_data(filepath):
    # Data loading and preprocessing as described in the report
    data = pd.read_csv(filepath)
    
    # Remove non-polypharmacy records (less than 5 medications)
    data['poly'] = data.iloc[:, 1:21].sum(axis=1)
    data = data[data['poly'] >= 5]
    
    # Remove redundant records
    data = data.drop_duplicates(subset=['patient_id', 'hospit'] + list(data.columns[1:21]))
    
    # Prepare features and target
    X = data.iloc[:, 1:21].values
    y = data['hospit'].values
    
    return X, y

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(test_loader), all_preds, all_labels



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Example usage of the configuration
    random_seed = cfg.random_seed
    test_size = cfg.test_size
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs
    k_folds = cfg.k_folds


    # MLflow tracking
    mlflow.set_experiment("Polypharmacy_Hospitalization_Prediction")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data('./PolypharmacyData.csv')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cross-validation setup
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Store cross-validation results
    cv_losses = []
    cv_accuracies = []
    
    # Cross-validation loop
    for fold, (train_index, val_index) in enumerate(kf.split(X_scaled)):
        with mlflow.start_run(nested=True):
            # Split data
            X_train, X_val = X_scaled[train_index], X_scaled[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Create PyTorch datasets
            train_dataset = PolypharmacyDataset(X_train, y_train)
            val_dataset = PolypharmacyDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            model = NeuralNetwork(input_size=X_train.shape[1]).to(device)
            
            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            for epoch in range(epochs):
                train_loss = train_model(model, train_loader, criterion, optimizer, device)
                val_loss, val_preds, val_labels = evaluate_model(model, val_loader, criterion, device)
                
                # Calculate accuracy
                val_accuracy = np.mean(np.array(val_preds) == np.array(val_labels))
                
                # MLflow logging
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, step=epoch)
            
            # Log fold results
            cv_losses.append(val_loss)
            cv_accuracies.append(val_accuracy)
            
            # Save model
            mlflow.pytorch.log_model(model, f"model_fold_{fold}")
    
    # Print cross-validation results
    print(f"Cross-validation Results:")
    print(f"Average Loss: {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")
    print(f"Average Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
    
    # Visualization of training
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(cv_losses, label='Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()