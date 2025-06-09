"""
Hybrid Intelligence Framework
Main implementation combining T-KMeans-LOF, IADAF, and LDPC modules
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from .preprocessing import TKMeansLOF
from .augmentation import IADAF
from .constraints import LDPCConstraints


class DNNModel(nn.Module):
    """
    Deep Neural Network for solubility prediction
    
    Architecture with flexible layer dimensions and 
    batch normalization for improved stability
    """
    
    def __init__(self, input_dim, output_dim, layer_dim=4, node_dim=64):
        """
        Parameters:
        -----------
        input_dim : int
            Number of input features
        output_dim : int
            Number of output features
        layer_dim : int
            Number of hidden layers
        node_dim : int
            Number of nodes per hidden layer
        """
        super(DNNModel, self).__init__()
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, node_dim),
            nn.BatchNorm1d(node_dim),
            nn.PReLU()
        ])
        
        # Hidden layers
        for _ in range(layer_dim - 2):
            layers.extend([
                nn.Linear(node_dim, node_dim),
                nn.BatchNorm1d(node_dim),
                nn.PReLU()
            ])
        
        # Output layer
        layers.extend([
            nn.Linear(node_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.PReLU()
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class HybridIntelligenceFramework:
    """
    Main framework integrating all three modules:
    - T-KMeans-LOF for outlier detection
    - IADAF for data augmentation  
    - LDPC for physical constraints
    """
    
    def __init__(self, input_cols, output_cols, binary_data_paths=None,
                 test_split=0.2, random_state=42):
        """
        Parameters:
        -----------
        input_cols : list
            Names of input feature columns
        output_cols : list
            Names of output feature columns
        binary_data_paths : dict
            Paths to binary system data for constraints
        test_split : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.test_split = test_split
        self.random_state = random_state
        
        # Initialize modules
        self.outlier_detector = TKMeansLOF()
        self.augmentor = IADAF()
        self.constraints = LDPCConstraints(binary_data_paths)
        
        # Model components
        self.model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training history
        self.history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        
    def preprocess_data(self, data):
        """
        Apply T-KMeans-LOF outlier detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw experimental data
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data with outliers removed
        """
        print("Step 1: Outlier Detection with T-KMeans-LOF")
        cleaned_data = self.outlier_detector.fit_transform(data)
        
        return cleaned_data
    
    def augment_data(self, data):
        """
        Apply IADAF data augmentation
        
        Parameters:
        -----------
        data : pd.DataFrame
            Cleaned data
            
        Returns:
        --------
        pd.DataFrame
            Augmented dataset
        """
        print("\nStep 2: Data Augmentation with IADAF")
        augmented_data = self.augmentor.fit_generate(data)
        
        return augmented_data
    
    def prepare_training_data(self, data):
        """
        Prepare data for neural network training
        
        Parameters:
        -----------
        data : pd.DataFrame
            Augmented dataset
            
        Returns:
        --------
        tuple
            Training and validation data splits
        """
        # Extract features and targets
        X = data[self.input_cols].values
        y = data[self.output_cols].values
        
        # Normalize data
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, 
            test_size=self.test_split,
            random_state=self.random_state
        )
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, data, layer_dim=4, node_dim=64, 
                    epochs=1000, learning_rate=0.008, batch_size=64):
        """
        Train DNN model on augmented data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Augmented dataset
        layer_dim : int
            Number of hidden layers
        node_dim : int
            Nodes per hidden layer
        epochs : int
            Training epochs
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        float
            Best validation loss achieved
        """
        print("\nStep 3: Training DNN Model")
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_training_data(data)
        
        # Initialize model
        self.model = DNNModel(
            input_dim=len(self.input_cols),
            output_dim=len(self.output_cols),
            layer_dim=layer_dim,
            node_dim=node_dim
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            
            train_pred = self.model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor)
            
            # Record history
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss.item())
            self.history['val_loss'].append(val_loss.item())
            
            # Save best model
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                self.best_model_state = self.model.state_dict()
                
            # Progress reporting
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
    
    def predict(self, X, apply_constraints=True):
        """
        Make predictions with optional physical constraints
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features [temperature, component1]
        apply_constraints : bool
            Whether to apply LDPC constraints
            
        Returns:
        --------
        array-like
            Predicted solubility values
        """
        # Ensure model is loaded
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X[self.input_cols].values
            
        # Scale input
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_scaled)
        
        # Apply constraints if requested
        if apply_constraints:
            # Combine input and predictions for constraint application
            full_predictions = np.hstack((X, y_pred))
            
            # Apply LDPC constraints
            constrained_predictions = self.constraints.apply(
                full_predictions, 
                self.model,
                self.scaler_x,
                self.scaler_y,
                self.device
            )
            
            # Extract only the prediction columns
            y_pred = constrained_predictions[:, len(self.input_cols):]
        
        return y_pred
    
    def save_model(self, path_prefix='model'):
        """
        Save trained model and scalers
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for saved file names
        """
        # Save model
        torch.save(self.model.state_dict(), f'{path_prefix}_dnn.pth')
        
        # Save scalers
        joblib.dump(self.scaler_x, f'{path_prefix}_scaler_x.pkl')
        joblib.dump(self.scaler_y, f'{path_prefix}_scaler_y.pkl')
        
        # Save configuration
        config = {
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'model_params': {
                'layer_dim': 4,
                'node_dim': 64
            }
        }
        joblib.dump(config, f'{path_prefix}_config.pkl')
        
        print(f"Model saved with prefix: {path_prefix}")
    
    def load_model(self, path_prefix='model'):
        """
        Load saved model and scalers
        
        Parameters:
        -----------
        path_prefix : str
            Prefix for saved file names
        """
        # Load configuration
        config = joblib.load(f'{path_prefix}_config.pkl')
        
        # Initialize model
        self.model = DNNModel(
            input_dim=len(config['input_cols']),
            output_dim=len(config['output_cols']),
            **config['model_params']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(f'{path_prefix}_dnn.pth'))
        
        # Load scalers
        self.scaler_x = joblib.load(f'{path_prefix}_scaler_x.pkl')
        self.scaler_y = joblib.load(f'{path_prefix}_scaler_y.pkl')
        
        print(f"Model loaded from: {path_prefix}")
    
    def fit(self, raw_data, **training_params):
        """
        Complete pipeline: preprocessing -> augmentation -> training
        
        Parameters:
        -----------
        raw_data : pd.DataFrame
            Raw experimental data
        **training_params : dict
            Parameters for model training
            
        Returns:
        --------
        self
            Fitted framework instance
        """
        # Step 1: Outlier detection
        cleaned_data = self.preprocess_data(raw_data)
        
        # Step 2: Data augmentation
        augmented_data = self.augment_data(cleaned_data)
        
        # Step 3: Model training
        self.train_model(augmented_data, **training_params)
        
        return self