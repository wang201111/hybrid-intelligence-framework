"""
Baseline ablation experiment for viscosity system with k-fold cross-validation.

This experiment serves as the baseline comparison by training a DNN directly
on real training data without any data augmentation. It evaluates the basic
DNN performance on the viscosity prediction task.

Experimental Design:
    1. Load ternary viscosity data (low temperature) and test data (high temperature)
    2. Create k-fold splits on training data (default: 5 folds)
    3. For each fold:
        - Train DNN on fold's training set
        - Validate on fold's validation set
        - Test on global test set (temperature extrapolation)
        - Physics evaluation with 3 binary boundary models
    4. Aggregate results across all folds

Viscosity System Characteristics:
    - Input: 4D [Temperature, Pressure, MCH%, cis-Decalin%]
    - Output: 1D [Viscosity]
    - Components: MCH (methylcyclohexane), cis-Decalin, HMN (heptamethylnonane)
    - Binary boundaries: 3 edges (MCH-HMN, MCH-Dec, Dec-HMN)
    - Physics evaluation: 3-boundary consistency + 4D Laplacian smoothness
"""

# Standard library imports
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# Project Root Detection
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from binary_predictor import BinaryPredictor
from utils_viscosity import ViscosityPhysicsEvaluator

import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# Logging Configuration
# ==============================================================================

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


# ==============================================================================
# Binary Predictor Wrapper for Physics Evaluation
# ==============================================================================

class BinaryPredictorWrapper:
    """
    Wrapper for BinaryPredictor to provide predict_physical interface.
    
    Converts 4D input [T, P, MCH, Dec] to 3D input [T, P, component]
    for binary boundary models.
    
    Attributes:
        predictor: BinaryPredictor instance.
        component_index: Which column to use as component (2=MCH, 3=Dec).
    """
    
    def __init__(self, predictor: BinaryPredictor, component_index: int = 2):
        """
        Initialize wrapper.
        
        Args:
            predictor: BinaryPredictor instance.
            component_index: Column index for component (2=MCH, 3=Dec).
        """
        self.predictor = predictor
        self.component_index = component_index
    
    def predict_physical(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with 4D to 3D conversion.
        
        Args:
            X: Input array [N, 4] - [T, P, MCH, Dec].
        
        Returns:
            Predictions [N, 1] - Viscosity.
        """
        # Extract 3D input: [T, P, component]
        X_3d = np.column_stack([
            X[:, 0],  # T
            X[:, 1],  # P
            X[:, self.component_index]  # MCH or Dec
        ])
        
        # Call BinaryPredictor.predict()
        y_pred = self.predictor.predict(X_3d)
        
        return y_pred
    
    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """Compatibility interface."""
        return self.predict_physical(X)


# ==============================================================================
# Utility Functions
# ==============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (can be dict, list, numpy type, or native type).
    
    Returns:
        Object with all numpy types converted to Python native types.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# ==============================================================================
# Data Loading
# ==============================================================================

def load_viscosity_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ternary viscosity data from Excel file.
    
    Args:
        filepath: Path to Excel file.
    
    Returns:
        Tuple of (X, y) where:
            - X: Features [N, 4] - [T, P, MCH%, cis-Decalin%]
            - y: Targets [N] - Viscosity
    
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file has insufficient columns.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_excel(filepath, engine='openpyxl')
    
    if len(df.columns) < 5:
        raise ValueError(
            f"Data file must have at least 5 columns (T, P, MCH, Dec, Viscosity), "
            f"found {len(df.columns)}"
        )
    
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(np.float32)
    
    logger.info(f"Loaded {filepath.name}:")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  T range: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}] °C")
    logger.info(f"  P range: [{X[:, 1].min():.2e}, {X[:, 1].max():.2e}] Pa")
    logger.info(f"  MCH range: [{X[:, 2].min():.2f}, {X[:, 2].max():.2f}] %")
    logger.info(f"  cis-Decalin range: [{X[:, 3].min():.2f}, {X[:, 3].max():.2f}] %")
    logger.info(f"  Viscosity range: [{y.min():.4f}, {y.max():.4f}] mPa·s")
    
    return X, y


def create_k_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create k-fold splits for cross-validation.
    
    Args:
        X: Feature array.
        y: Target array.
        n_splits: Number of folds.
        random_state: Random seed for reproducibility.
    
    Returns:
        List of dictionaries, each containing:
            - X_train: Training features
            - y_train: Training targets
            - X_val: Validation features
            - y_val: Validation targets
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in kfold.split(X):
        fold = {
            'X_train': X[train_idx].copy(),
            'y_train': y[train_idx].copy(),
            'X_val': X[val_idx].copy(),
            'y_val': y[val_idx].copy()
        }
        folds.append(fold)
    
    return folds


# ==============================================================================
# DNN Model
# ==============================================================================

class DNN(nn.Module):
    """
    Deep neural network for viscosity prediction.
    
    Attributes:
        input_dim: Input feature dimension (4 for viscosity).
        hidden_dim: Hidden layer dimension.
        layer_dim: Number of hidden layers.
        output_dim: Output dimension (1 for viscosity).
        layers: Sequential network layers.
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        layer_dim: int = 4,
        output_dim: int = 1
    ):
        """
        Initialize DNN model.
        
        Args:
            input_dim: Number of input features.
            hidden_dim: Number of hidden units per layer.
            layer_dim: Number of hidden layers.
            output_dim: Number of output features.
        """
        super(DNN, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(layer_dim - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim].
        
        Returns:
            Output tensor [batch_size, output_dim].
        """
        return self.layers(x)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class BaselineConfig:
    """
    Configuration for baseline viscosity experiment.
    
    Attributes:
        data_dir: Base data directory.
        low_temp_file: Low temperature data file (for k-fold CV).
        high_temp_file: High temperature data file (for extrapolation test).
        models_dir: Directory containing binary boundary models.
        mch_hmn_model_file: MCH-HMN binary model filename.
        mch_dec_model_file: MCH-cis-Decalin binary model filename.
        dec_hmn_model_file: cis-Decalin-HMN binary model filename.
        k_folds: Number of folds for cross-validation.
        kfold_random_state: Random seed for k-fold splitting.
        dnn_epochs: Number of training epochs.
        dnn_learning_rate: Learning rate.
        dnn_layer_dim: Number of DNN layers.
        dnn_node_dim: Number of nodes per layer.
        boundary_decay_lambda: Decay coefficient for boundary scoring.
        smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        output_dir: Output directory for results.
        save_predictions: Whether to save predictions to Excel.
        save_metrics: Whether to save metrics to Excel.
        excel_prefix: Prefix for Excel output files.
        random_seed: Random seed for reproducibility.
        device: Computing device ('auto', 'cuda', 'cpu').
        log_level: Logging level.
    """
    data_dir: Path = PROJECT_ROOT / 'data' / 'viscosity' / 'split_by_temperature'
    low_temp_file: str = 'low_temp.xlsx'
    high_temp_file: str = 'high_temp.xlsx'
    
    models_dir: Path = PROJECT_ROOT / 'models' / 'viscosity' / 'binary'
    mch_hmn_model_file: str = 'MCH_HMN.pth'
    mch_dec_model_file: str = 'MCH_cis_Decalin.pth'
    dec_hmn_model_file: str = 'cis_Decalin_HMN.pth'
    
    k_folds: int = 5
    kfold_random_state: int = 42
    
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    output_dir: Path = PROJECT_ROOT / 'results' / 'viscosity' / 'ablation' / 'baseline_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'baseline_'
    
    random_seed: int = 42
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: auto-detect device."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# Metrics Saving
# ==============================================================================

def save_fold_metrics_to_excel(
    metrics_dict: Dict[str, Any],
    physics_results: Optional[Dict[str, Any]],
    save_path: Path,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Save fold metrics to Excel file.
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics.
        physics_results: Optional physics evaluation results.
        save_path: Path to save Excel file.
        logger_instance: Optional logger instance.
    """
    if logger_instance is None:
        logger_instance = logger
    
    data = []
    
    data.append(['Train RMSE', metrics_dict.get('train_rmse', float('nan'))])
    data.append(['Train MAE', metrics_dict.get('train_mae', float('nan'))])
    data.append(['Train R²', metrics_dict.get('train_r2', float('nan'))])
    
    data.append(['Val RMSE', metrics_dict.get('val_rmse', float('nan'))])
    data.append(['Val MAE', metrics_dict.get('val_mae', float('nan'))])
    data.append(['Val R²', metrics_dict.get('val_r2', float('nan'))])
    
    data.append(['Test RMSE', metrics_dict.get('test_rmse', float('nan'))])
    data.append(['Test MAE', metrics_dict.get('test_mae', float('nan'))])
    data.append(['Test R²', metrics_dict.get('test_r2', float('nan'))])
    
    if physics_results is not None:
        physics_total = physics_results.get('overall_score', float('nan'))
        data.append(['Physics Score', physics_total])
        
        physics_boundary = physics_results.get('boundary_score', float('nan'))
        data.append(['Boundary Consistency', physics_boundary])
        
        physics_smoothness = physics_results.get('smoothness_score', float('nan'))
        data.append(['Thermodynamic Smoothness', physics_smoothness])
        
        logger_instance.info(
            f"Physics metrics included: overall={physics_total:.4f}, "
            f"boundary={physics_boundary:.4f}, smoothness={physics_smoothness:.4f}"
        )
    else:
        logger_instance.warning("Physics results is None - no physics metrics will be saved")
    
    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    logger_instance.info(f"Metrics saved to: {save_path.name}")


def save_predictions_to_excel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    save_dir: Path,
    prefix: str = ''
) -> None:
    """
    Save predictions to Excel files.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        y_train_pred: Training predictions.
        X_val: Validation features.
        y_val: Validation targets.
        y_val_pred: Validation predictions.
        X_test: Test features.
        y_test: Test targets.
        y_test_pred: Test predictions.
        save_dir: Directory to save files.
        prefix: Filename prefix.
    """
    excel_dir = save_dir / 'excel'
    excel_dir.mkdir(parents=True, exist_ok=True)
    
    df_test = pd.DataFrame({
        'Temperature': X_test[:, 0],
        'Pressure': X_test[:, 1],
        'MCH': X_test[:, 2],
        'cis_Decalin': X_test[:, 3],
        'Viscosity_True': y_test.flatten(),
        'Viscosity_Pred': y_test_pred.flatten(),
        'Error': y_test.flatten() - y_test_pred.flatten()
    })
    df_test.to_excel(excel_dir / f'{prefix}test_predictions.xlsx', index=False)
    
    df_train = pd.DataFrame({
        'Temperature': X_train[:, 0],
        'Pressure': X_train[:, 1],
        'MCH': X_train[:, 2],
        'cis_Decalin': X_train[:, 3],
        'Viscosity_True': y_train.flatten(),
        'Viscosity_Pred': y_train_pred.flatten(),
        'Error': y_train.flatten() - y_train_pred.flatten()
    })
    df_train.to_excel(excel_dir / f'{prefix}train_predictions.xlsx', index=False)
    
    df_val = pd.DataFrame({
        'Temperature': X_val[:, 0],
        'Pressure': X_val[:, 1],
        'MCH': X_val[:, 2],
        'cis_Decalin': X_val[:, 3],
        'Viscosity_True': y_val.flatten(),
        'Viscosity_Pred': y_val_pred.flatten(),
        'Error': y_val.flatten() - y_val_pred.flatten()
    })
    df_val.to_excel(excel_dir / f'{prefix}val_predictions.xlsx', index=False)


# ==============================================================================
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    physics_results: Optional[Dict[str, Any]],
    config: BaselineConfig,
    save_dir: Path,
    fold_idx: int
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics.
        physics_results: Physics evaluation results.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        fold_idx: Fold index.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Baseline Ablation - Viscosity System',
            'fold_index': fold_idx,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device
        },
        'model_architecture': {
            'type': 'Baseline DNN (no augmentation)',
            'epochs': config.dnn_epochs,
            'learning_rate': config.dnn_learning_rate,
            'layer_dim': config.dnn_layer_dim,
            'node_dim': config.dnn_node_dim
        },
        'k_fold_config': {
            'n_folds': config.k_folds,
            'random_state': config.kfold_random_state,
            'current_fold': fold_idx
        },
        'physics_config': {
            'boundary_decay_lambda': config.boundary_decay_lambda,
            'smoothness_decay_lambda': config.smoothness_decay_lambda
        },
        'metrics': metrics
    }
    
    if physics_results is not None:
        config_dict['physics_results'] = {
            'overall_score': physics_results.get('overall_score'),
            'boundary_score': physics_results.get('boundary_score'),
            'smoothness_score': physics_results.get('smoothness_score')
        }
    
    config_dict = convert_numpy_types(config_dict)
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


# ==============================================================================
# Training Function
# ==============================================================================

def train_baseline_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: BaselineConfig,
    save_dir: Path
) -> Dict[str, Any]:
    """
    Train baseline DNN model.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        X_test: Test features.
        y_test: Test targets.
        config: Experiment configuration.
        save_dir: Directory to save results.
    
    Returns:
        Dictionary containing:
            - model: Trained model
            - scalers: (x_scaler, y_scaler)
            - metrics: Evaluation metrics
            - predictions: Model predictions
            - true_values: True target values
    """
    device = torch.device(config.device)
    
    logger.info("="*70)
    logger.info("Training Baseline DNN")
    logger.info("="*70)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    X_test_scaled = x_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    model = DNN(
        input_dim=4,
        hidden_dim=config.dnn_node_dim,
        layer_dim=config.dnn_layer_dim,
        output_dim=1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.dnn_learning_rate)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
    
    logger.info(f"Training for {config.dnn_epochs} epochs...")
    
    model.train()
    for epoch in tqdm(range(config.dnn_epochs), desc="Training"):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    logger.info("Training completed")
    
    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(X_train_tensor).cpu().numpy().flatten()
        y_val_pred_scaled = model(X_val_tensor).cpu().numpy().flatten()
        
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()
    
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    
    metrics = {
        'train_r2': sklearn_metrics.r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_train, y_train_pred)),
        'train_mae': sklearn_metrics.mean_absolute_error(y_train, y_train_pred),
        'val_r2': sklearn_metrics.r2_score(y_val, y_val_pred),
        'val_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_val, y_val_pred)),
        'val_mae': sklearn_metrics.mean_absolute_error(y_val, y_val_pred),
        'test_r2': sklearn_metrics.r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_test, y_test_pred)),
        'test_mae': sklearn_metrics.mean_absolute_error(y_test, y_test_pred)
    }
    
    logger.info("="*70)
    logger.info("Evaluation Results")
    logger.info("="*70)
    logger.info(f"Train R²:  {metrics['train_r2']:.6f}")
    logger.info(f"Val R²:    {metrics['val_r2']:.6f}")
    logger.info(f"Test R²:   {metrics['test_r2']:.6f}")
    logger.info(f"Train RMSE: {metrics['train_rmse']:.6f}")
    logger.info(f"Val RMSE:   {metrics['val_rmse']:.6f}")
    logger.info(f"Test RMSE:  {metrics['test_rmse']:.6f}")
    logger.info("="*70)
    
    model_path = save_dir / 'models' / 'baseline_model.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'config': {
            'input_dim': 4,
            'hidden_dim': config.dnn_node_dim,
            'layer_dim': config.dnn_layer_dim,
            'output_dim': 1
        }
    }, model_path)
    
    return {
        'model': model,
        'scalers': (x_scaler, y_scaler),
        'metrics': metrics,
        'predictions': {
            'train': y_train_pred,
            'val': y_val_pred,
            'test': y_test_pred
        },
        'true_values': {
            'train': y_train,
            'val': y_val,
            'test': y_test
        }
    }


# ==============================================================================
# Physics Evaluation
# ==============================================================================

def run_physics_evaluation(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    config: BaselineConfig,
    save_dir: Path,
    device: str
) -> Optional[Dict[str, Any]]:
    """
    Run physics consistency evaluation.
    
    Args:
        model: Trained DNN model.
        x_scaler: Feature scaler.
        y_scaler: Target scaler.
        config: Experiment configuration.
        save_dir: Directory to save physics evaluation report.
        device: Computing device.
    
    Returns:
        Dictionary with physics evaluation results, or None if models unavailable.
    """
    logger.info("")
    logger.info("="*70)
    logger.info("Physics Consistency Evaluation")
    logger.info("="*70)
    
    mch_hmn_path = config.models_dir / config.mch_hmn_model_file
    mch_dec_path = config.models_dir / config.mch_dec_model_file
    dec_hmn_path = config.models_dir / config.dec_hmn_model_file
    
    logger.info("Loading binary boundary models...")
    logger.info(f"  MCH-HMN: {mch_hmn_path}")
    logger.info(f"  MCH-Dec: {mch_dec_path}")
    logger.info(f"  Dec-HMN: {dec_hmn_path}")
    
    try:
        if not mch_hmn_path.exists():
            raise FileNotFoundError(f"MCH-HMN model not found: {mch_hmn_path}")
        if not mch_dec_path.exists():
            raise FileNotFoundError(f"MCH-Dec model not found: {mch_dec_path}")
        if not dec_hmn_path.exists():
            raise FileNotFoundError(f"Dec-HMN model not found: {dec_hmn_path}")
        
        teacher_mch_hmn_raw = BinaryPredictor.load(str(mch_hmn_path))
        teacher_mch_dec_raw = BinaryPredictor.load(str(mch_dec_path))
        teacher_dec_hmn_raw = BinaryPredictor.load(str(dec_hmn_path))
        
        logger.info("All binary models loaded successfully")
        
        # Wrap teachers to provide predict_physical interface
        # MCH-HMN (Dec=0 boundary): component_index=2 (MCH)
        teacher_mch_hmn = BinaryPredictorWrapper(teacher_mch_hmn_raw, component_index=2)
        
        # Dec-HMN (MCH=0 boundary): component_index=3 (Dec) 
        teacher_dec_hmn = BinaryPredictorWrapper(teacher_dec_hmn_raw, component_index=3)
        
        # MCH-Dec (HMN=0 boundary): component_index=2 (MCH)
        teacher_mch_dec = BinaryPredictorWrapper(teacher_mch_dec_raw, component_index=2)
        
        logger.info("Binary models wrapped with predict_physical interface")
        
        evaluator = ViscosityPhysicsEvaluator(
            teacher_models=(teacher_mch_hmn, teacher_dec_hmn, teacher_mch_dec),
            temp_range=(20.0, 80.0),
            pressure_range=(1e5, 1e8),
            boundary_decay_lambda=config.boundary_decay_lambda,
            smoothness_decay_lambda=config.smoothness_decay_lambda,
            log_level=config.log_level
        )
        
        class ModelWrapper:
            """Wrapper to make DNN compatible with evaluator."""
            def __init__(self, model, x_scaler, y_scaler, device):
                self.model = model
                self.x_scaler = x_scaler
                self.y_scaler = y_scaler
                self.device = device
            
            def predict(self, X, return_original_scale=True):
                X_scaled = self.x_scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    y_pred_scaled = self.model(X_tensor).cpu().numpy().flatten()
                
                if return_original_scale:
                    y_pred = self.y_scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).flatten()
                    return y_pred
                else:
                    return y_pred_scaled
        
        trainer = ModelWrapper(model, x_scaler, y_scaler, device)
        
        overall_score, results = evaluator.evaluate_full(trainer)
        
        logger.info("")
        logger.info("="*70)
        logger.info("Physics Evaluation Complete")
        logger.info("="*70)
        logger.info(f"Overall Physics Score: {overall_score:.6f}")
        logger.info(f"  Boundary Score: {results['boundary_score']:.6f}")
        logger.info(f"  Smoothness Score: {results['smoothness_score']:.6f}")
        logger.info("="*70)
        
        report = evaluator.generate_evaluation_report(results)
        report_path = save_dir / 'data' / 'physics_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Physics evaluation report saved: {report_path}")
        
        return results
        
    except Exception as e:
        import traceback
        logger.error("")
        logger.error("="*70)
        logger.error("Physics Evaluation FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error("")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        logger.error("")
        logger.error("Continuing without physics evaluation...")
        logger.error("="*70)
        return None


# ==============================================================================
# K-Fold Experiment Manager
# ==============================================================================

class KFoldExperimentManager:
    """
    Manager for k-fold cross-validation baseline experiment.
    
    Coordinates multiple folds of the baseline experiment:
    - Loads data and creates k-fold splits
    - Runs baseline DNN training for each fold
    - Physics evaluation for each fold
    - Aggregates results across folds
    
    Attributes:
        config: Experiment configuration.
        n_folds: Number of folds.
        output_dir: Main output directory.
        all_results: List storing results from all folds.
        logger: Logger instance.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.n_folds = config.k_folds
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = config.output_dir / f'run_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results: List[Dict[str, Any]] = []
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def run_all_folds(self) -> None:
        """Run experiments for all folds."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"K-Fold Baseline Experiment - Viscosity System ({self.n_folds} folds)")
        self.logger.info("="*70)
        
        total_start = time.time()
        
        low_temp_path = self.config.data_dir / self.config.low_temp_file
        high_temp_path = self.config.data_dir / self.config.high_temp_file
        
        self.logger.info("")
        self.logger.info("Loading data and creating k-fold splits...")
        
        X_low_temp, y_low_temp = load_viscosity_data(low_temp_path)
        X_test, y_test = load_viscosity_data(high_temp_path)
        
        folds = create_k_folds(
            X_low_temp,
            y_low_temp,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state
        )
        
        self.logger.info(f"K-fold splits created (n_splits={self.n_folds})")
        for i, fold in enumerate(folds):
            self.logger.info(
                f"  Fold {i}: Train={len(fold['X_train'])}, Val={len(fold['X_val'])}"
            )
        
        for fold_idx, fold_data in enumerate(folds):
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Fold {fold_idx} Experiment")
            self.logger.info("="*70)
            
            fold_dir = self.output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)
            
            X_train = fold_data['X_train']
            y_train = fold_data['y_train']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            
            results = train_baseline_dnn(
                X_train, y_train, X_val, y_val, X_test, y_test,
                self.config, fold_dir
            )
            
            if self.config.save_predictions:
                save_predictions_to_excel(
                    X_train, y_train, results['predictions']['train'],
                    X_val, y_val, results['predictions']['val'],
                    X_test, y_test, results['predictions']['test'],
                    fold_dir,
                    prefix=self.config.excel_prefix
                )
            
            physics_results = run_physics_evaluation(
                results['model'],
                results['scalers'][0],
                results['scalers'][1],
                self.config,
                fold_dir,
                self.config.device
            )
            
            if physics_results is None:
                self.logger.warning("")
                self.logger.warning("="*70)
                self.logger.warning(f"Fold {fold_idx}: Physics evaluation returned None")
                self.logger.warning("Physics scores will NOT be saved in metrics Excel file")
                self.logger.warning("Check if binary model files exist at:")
                self.logger.warning(f"  - {self.config.models_dir / self.config.mch_hmn_model_file}")
                self.logger.warning(f"  - {self.config.models_dir / self.config.mch_dec_model_file}")
                self.logger.warning(f"  - {self.config.models_dir / self.config.dec_hmn_model_file}")
                self.logger.warning("="*70)
            else:
                self.logger.info(f"Fold {fold_idx}: Physics evaluation successful")
                self.logger.info(f"  Overall Score: {physics_results['overall_score']:.6f}")
                self.logger.info(f"  Boundary Score: {physics_results['boundary_score']:.6f}")
                self.logger.info(f"  Smoothness Score: {physics_results['smoothness_score']:.6f}")
            
            if self.config.save_metrics:
                save_fold_metrics_to_excel(
                    results['metrics'],
                    physics_results,
                    fold_dir / 'excel' / f'{self.config.excel_prefix}metrics.xlsx',
                    self.logger
                )
            
            save_config(
                results['metrics'],
                physics_results,
                self.config,
                fold_dir,
                fold_idx
            )
            
            result_dict = {
                'fold_idx': fold_idx,
                'metrics': results['metrics'],
                'physics_results': physics_results
            }
            self.all_results.append(result_dict)
            
            self.logger.info(f"Fold {fold_idx} completed")
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Generating summary...")
        self.logger.info("="*70)
        
        self.generate_summary()
        
        total_time = time.time() - total_start
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("All experiments completed!")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def generate_summary(self) -> None:
        """Generate summary across all folds."""
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        metrics_data = []
        
        for result in self.all_results:
            fold_idx = result['fold_idx']
            metrics = result['metrics']
            physics = result.get('physics_results')
            
            row = {
                'Fold': fold_idx,
                'Train_R2': metrics['train_r2'],
                'Val_R2': metrics['val_r2'],
                'Test_R2': metrics['test_r2'],
                'Train_RMSE': metrics['train_rmse'],
                'Val_RMSE': metrics['val_rmse'],
                'Test_RMSE': metrics['test_rmse'],
                'Train_MAE': metrics['train_mae'],
                'Val_MAE': metrics['val_mae'],
                'Test_MAE': metrics['test_mae']
            }
            
            if physics is not None:
                row['Physics_Overall'] = physics.get('overall_score', np.nan)
                row['Physics_Boundary'] = physics.get('boundary_score', np.nan)
                row['Physics_Smoothness'] = physics.get('smoothness_score', np.nan)
            else:
                row['Physics_Overall'] = np.nan
                row['Physics_Boundary'] = np.nan
                row['Physics_Smoothness'] = np.nan
            
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        summary_stats = df.describe().T
        summary_stats.to_excel(summary_dir / 'summary_statistics.xlsx')
        
        df.to_excel(summary_dir / 'all_folds_metrics.xlsx', index=False)
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("K-Fold Baseline Experiment Summary - Viscosity System")
        report_lines.append("="*70)
        report_lines.append(f"\nNumber of folds: {self.n_folds}")
        report_lines.append(f"Random state: {self.config.kfold_random_state}")
        report_lines.append("\n" + "-"*70)
        report_lines.append("Test Set Performance (Mean ± Std)")
        report_lines.append("-"*70)
        
        test_r2_mean = df['Test_R2'].mean()
        test_r2_std = df['Test_R2'].std()
        test_rmse_mean = df['Test_RMSE'].mean()
        test_rmse_std = df['Test_RMSE'].std()
        
        report_lines.append(f"Test R²:   {test_r2_mean:.6f} ± {test_r2_std:.6f}")
        report_lines.append(f"Test RMSE: {test_rmse_mean:.6f} ± {test_rmse_std:.6f}")
        
        if not df['Physics_Overall'].isna().all():
            phys_mean = df['Physics_Overall'].mean()
            phys_std = df['Physics_Overall'].std()
            report_lines.append(f"Physics Score: {phys_mean:.6f} ± {phys_std:.6f}")
        
        report_lines.append("\n" + "="*70)
        
        report_text = "\n".join(report_lines)
        
        with open(summary_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        self.logger.info("\n" + report_text)


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """
    Main entry point for baseline viscosity experiment.
    
    Configures and runs k-fold cross-validation baseline experiment.
    """
    config = BaselineConfig()
    
    config.k_folds = 5
    config.kfold_random_state = 42
    
    config.dnn_epochs = 1000
    config.dnn_learning_rate = 0.00831
    config.dnn_layer_dim = 4
    config.dnn_node_dim = 128
    
    config.boundary_decay_lambda = 5.0
    config.smoothness_decay_lambda = 15.0
    
    config.save_predictions = True
    config.save_metrics = True
    config.log_level = logging.INFO
    
    logger.info("")
    logger.info("="*70)
    logger.info("Project Paths")
    logger.info("="*70)
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"  - Low temp: {config.data_dir / config.low_temp_file}")
    logger.info(f"  - High temp: {config.data_dir / config.high_temp_file}")
    logger.info(f"Models directory: {config.models_dir}")
    logger.info(f"  - MCH-HMN: {config.models_dir / config.mch_hmn_model_file}")
    logger.info(f"  - MCH-Dec: {config.models_dir / config.mch_dec_model_file}")
    logger.info(f"  - Dec-HMN: {config.models_dir / config.dec_hmn_model_file}")
    logger.info(f"Output directory: {config.output_dir}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("Baseline Viscosity Experiment")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-folds: {config.k_folds}")
    logger.info(f"Random state: {config.kfold_random_state}")
    logger.info("")
    logger.info("Model configuration:")
    logger.info(f"  Epochs: {config.dnn_epochs}")
    logger.info(f"  Learning rate: {config.dnn_learning_rate}")
    logger.info(f"  Layers: {config.dnn_layer_dim}")
    logger.info(f"  Nodes per layer: {config.dnn_node_dim}")
    logger.info("")
    logger.info("Physics evaluation:")
    logger.info(f"  Boundary lambda: {config.boundary_decay_lambda}")
    logger.info(f"  Smoothness lambda: {config.smoothness_decay_lambda}")
    
    manager = KFoldExperimentManager(config)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("Baseline viscosity experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()