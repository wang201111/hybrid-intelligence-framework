"""
================================================================================
Small Sample Sensitivity Experiment - Baseline (Solubility System)
================================================================================

Evaluates baseline DNN performance sensitivity to training set size using
repeated random sampling without any data augmentation, outlier detection,
or physical constraints.

Key Features:
    - Multiple sampling fractions: 10%, 25%, 50%, 75%, 100%
    - Repeated random sampling (10 repeats per fraction)
    - Internal train/validation split for early stopping
    - Independent evaluation on fixed global validation and test sets
    - Complete metrics tracking and physics evaluation
    - No visualization code (pure evaluation)

Experimental Design:
    1. Sample fraction of training pool
    2. Internal split: 80% internal train, 20% internal validation
    3. Train DNN with early stopping on internal validation
    4. Evaluate on fixed global validation and test sets
    5. Repeat 10 times with different random seeds
    
Standardization:
    - PEP 8 compliant
    - Complete type hints (PEP 484)
    - Google-style docstrings (PEP 257)
    - Logging module instead of print
    - English documentation throughout
    - Robust path handling (works from any directory)
    - No hardcoded paths
    - No visualization code

File Location:
    experiments/solubility/small_sample/baseline.py
    
================================================================================
"""

# Standard library imports
import copy
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# Project Root Detection
# ==============================================================================

# File location: experiments/solubility/small_sample/baseline.py
# Need 3 parents to reach project root:
# baseline.py -> small_sample/ -> solubility/ -> experiments/ -> PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from tqdm import tqdm

# Local imports
from utils_solubility import (
    DNN,
    PhysicalConsistencyEvaluator,
    TSTREvaluator,
)
from binary_predictor import BinaryPredictor

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


# Initialize module-level logger
logger = get_logger(__name__)


# ==============================================================================
# Metrics Saving Function
# ==============================================================================

def save_fold_metrics_to_excel(
    metrics_dict: Dict[str, Any],
    physics_results: Optional[Dict[str, Any]],
    save_path: Path,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Save fold metrics to Excel file.
    
    Creates a two-column Excel file with metrics and their values:
        - Column 1: Metric name
        - Column 2: Metric value
    
    Includes both machine learning metrics (R², RMSE, MAE) and
    physics evaluation scores (boundary consistency, smoothness).
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics.
        physics_results: Optional physics evaluation results.
        save_path: Path to save Excel file.
        logger_instance: Optional logger instance.
    
    Examples:
        >>> save_fold_metrics_to_excel(
        ...     metrics_dict={'train_r2': 0.95, ...},
        ...     physics_results={'overall_score': 0.87, ...},
        ...     save_path=Path('./results/metrics.xlsx')
        ... )
    """
    if logger_instance is None:
        logger_instance = logger
    
    data = []
    
    # Extract metrics (handle both nested and flat structures)
    if 'metrics' in metrics_dict:
        inner_metrics = metrics_dict['metrics']
    else:
        inner_metrics = metrics_dict
    
    # Training set metrics
    data.append(['Train RMSE', inner_metrics.get('train_rmse', float('nan'))])
    data.append(['Train MAE', inner_metrics.get('train_mae', float('nan'))])
    data.append(['Train R²', inner_metrics.get('train_r2', float('nan'))])
    
    # Validation set metrics
    data.append(['Val RMSE', inner_metrics.get('val_rmse', float('nan'))])
    data.append(['Val MAE', inner_metrics.get('val_mae', float('nan'))])
    data.append(['Val R²', inner_metrics.get('val_r2', float('nan'))])
    
    # Test set metrics
    data.append(['Test RMSE', inner_metrics.get('test_rmse', float('nan'))])
    data.append(['Test MAE', inner_metrics.get('test_mae', float('nan'))])
    data.append(['Test R²', inner_metrics.get('test_r2', float('nan'))])
    
    # Physics evaluation scores (if available)
    if physics_results is not None:
        physics_total = physics_results.get('overall_score', float('nan'))
        data.append(['Physics Score', physics_total])
        
        physics_boundary = physics_results.get('boundary_score', float('nan'))
        data.append(['Boundary Consistency', physics_boundary])
        
        physics_smoothness = physics_results.get('smoothness_score', float('nan'))
        data.append(['Thermodynamic Smoothness', physics_smoothness])
        
        logger_instance.debug(f"Physics metrics included: overall={physics_total:.4f}, boundary={physics_boundary:.4f}, smoothness={physics_smoothness:.4f}")
    else:
        logger_instance.debug("No physics results available")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    
    # Save to Excel
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    logger_instance.debug(f"Metrics saved: {save_path.name}")


# ==============================================================================
# Data Loading Function
# ==============================================================================

def load_ternary_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ternary system data from Excel file.
    
    Args:
        filepath: Path to data file (.xlsx).
    
    Returns:
        Tuple of (X, y) where:
            - X: Feature array of shape (n_samples, 2) - [Temperature, Input_Component]
            - y: Target array of shape (n_samples,) - Output_Component
    
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = pd.read_excel(filepath, engine='openpyxl')
    
    col_names = data.columns.tolist()
    
    # Find temperature column
    temp_col = None
    for col in col_names:
        if 'T' in col or 'temp' in col.lower():
            temp_col = col
            break
    
    if temp_col is None:
        temp_col = col_names[0]
    
    comp_cols = [col for col in col_names if col != temp_col]
    
    if len(comp_cols) < 2:
        raise ValueError(f"Expected at least 2 composition columns, found {len(comp_cols)}")
    
    X = data[[temp_col, comp_cols[0]]].values.astype(np.float32)
    y = data[comp_cols[1]].values.astype(np.float32)
    
    return X, y


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class SmallSampleConfig:
    """
    Configuration for small sample sensitivity experiment.
    
    All paths are ABSOLUTE paths based on PROJECT_ROOT for reliability.
    
    Attributes:
        Data Paths (absolute paths):
            data_dir: Base data directory.
            train_pool_file: Training pool data filename.
            validation_file: Global validation data filename.
            test_file: Test data filename.
        
        Model Paths (absolute paths):
            models_dir: Directory containing boundary models.
            input_boundary_model_file: Input boundary model filename.
            output_boundary_model_file: Output boundary model filename.
        
        Binary Model Data Paths (absolute paths):
            binary_data_dir: Directory containing binary system data.
            input_boundary_data_file: Input boundary training data filename.
            output_boundary_data_file: Output boundary training data filename.
        
        Sampling Configuration:
            sample_fractions: List of sampling fractions to test.
            n_sampling_repeats: Number of random sampling repeats per fraction.
            internal_val_ratio: Ratio of sampled data used for internal validation.
        
        Training Parameters:
            dnn_epochs: Number of training epochs.
            dnn_learning_rate: Learning rate.
            dnn_layer_dim: Number of DNN layers.
            dnn_node_dim: Number of nodes per layer.
            early_stop_patience: Patience for early stopping.
            early_stop_start_epoch: Epoch to start early stopping.
        
        Random Seeds:
            training_seed: Fixed seed for model initialization.
            sampling_base_seed: Base seed for sampling (incremented per sample).
        
        Physics Evaluation:
            boundary_decay_lambda: Decay coefficient for boundary scoring.
            smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        
        Output Settings (absolute paths):
            output_dir: Output directory for results.
            save_predictions: Whether to save predictions to Excel.
            save_metrics: Whether to save metrics to Excel.
            excel_prefix: Prefix for Excel output files.
        
        Device:
            device: Computing device ('auto', 'cuda', 'cpu').
            log_level: Logging level.
    """
    # Data paths (absolute paths based on PROJECT_ROOT)
    data_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'fixed_splits'
    train_pool_file: str = 'train.xlsx'
    validation_file: str = 'val.xlsx'
    test_file: str = 'test.xlsx'
    
    # Model paths (absolute paths based on PROJECT_ROOT)
    models_dir: Path = PROJECT_ROOT / 'models' / 'solubility' / 'binary'
    input_boundary_model_file: str = 'KCl_H2O.pth'
    output_boundary_model_file: str = 'MgCl2_H2O.pth'
    
    # Binary model data paths (absolute paths based on PROJECT_ROOT)
    binary_data_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'raw' / 'binary'
    input_boundary_data_file: str = 'KCl_H2O.xlsx'
    output_boundary_data_file: str = 'MgCl2_H2O.xlsx'
    
    # Sampling configuration
    sample_fractions: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.00]
    )
    n_sampling_repeats: int = 10
    internal_val_ratio: float = 0.2
    
    # Training parameters
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.008311634679852062
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    early_stop_patience: int = 1000
    early_stop_start_epoch: int = 500
    
    # Random seeds
    training_seed: int = 42
    sampling_base_seed: int = 10000
    
    # Physics evaluation parameters
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    # Output settings (absolute path based on PROJECT_ROOT)
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'small_sample' / 'small_sample_baseline_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'baseline_'
    
    # Device
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: auto-detect device."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# Data Sampling Function
# ==============================================================================

def sample_and_split_internal(
    train_pool: Tuple[np.ndarray, np.ndarray],
    fraction: float,
    sampling_idx: int,
    internal_val_ratio: float = 0.2,
    base_seed: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, int]:
    """
    Sample from training pool and perform internal split.
    
    Strategy:
        1. Randomly sample 'fraction' of training pool
        2. Split sampled data into internal train (80%) and internal val (20%)
        3. Internal val is used only for early stopping
        4. Return both full sampled data and internal split
    
    Args:
        train_pool: Tuple of (X_pool, y_pool).
        fraction: Sampling fraction (0.0-1.0).
        sampling_idx: Sampling index for seed generation.
        internal_val_ratio: Ratio for internal validation split.
        base_seed: Base seed for reproducibility.
    
    Returns:
        Tuple of:
            - X_sampled_full: Full sampled features (for final evaluation).
            - y_sampled_full: Full sampled targets (for final evaluation).
            - X_internal_train: Internal training features.
            - y_internal_train: Internal training targets.
            - X_internal_val: Internal validation features.
            - y_internal_val: Internal validation targets.
            - sampling_seed: Actual seed used.
    
    Examples:
        >>> X_full, y_full, X_tr, y_tr, X_val, y_val, seed = sample_and_split_internal(
        ...     train_pool, fraction=0.5, sampling_idx=0
        ... )
    """
    X_pool, y_pool = train_pool
    
    # Generate unique seed for this sampling
    sampling_seed = base_seed + int(fraction * 1000) + sampling_idx * 100
    
    # Step 1: Random sampling from pool
    np.random.seed(sampling_seed)
    n_total = len(X_pool)
    n_sample = int(n_total * fraction)
    
    indices = np.random.choice(n_total, size=n_sample, replace=False)
    X_subset = X_pool[indices].copy()
    y_subset = y_pool[indices].copy()
    
    # Save full sampled data
    X_sampled_full = X_subset.copy()
    y_sampled_full = y_subset.copy()
    
    # Step 2: Internal split for early stopping
    split_seed = sampling_seed + 1
    
    X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
        X_subset, y_subset,
        test_size=internal_val_ratio,
        random_state=split_seed
    )
    
    return (X_sampled_full, y_sampled_full, 
            X_internal_train, y_internal_train, 
            X_internal_val, y_internal_val, 
            sampling_seed)


# ==============================================================================
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    config: SmallSampleConfig,
    save_dir: Path,
    sampling_info: Dict[str, Any]
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        sampling_info: Sampling information dictionary.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Small Sample Sensitivity Study (Baseline)',
            'version': 'v3.0 (Standardized)',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device,
            'sampling_fraction': sampling_info['fraction'],
            'sampling_index': sampling_info['sampling_idx'],
            'sampling_seed': sampling_info['sampling_seed']
        },
        'data_info': {
            'n_sampled_full': sampling_info['n_sampled_full'],
            'n_internal_train': sampling_info['n_internal_train'],
            'n_internal_val': sampling_info['n_internal_val'],
            'n_global_val': sampling_info.get('n_global_val', 'N/A'),
            'n_test': sampling_info.get('n_test', 'N/A'),
            'internal_val_ratio': config.internal_val_ratio
        },
        'training_config': {
            'dnn_epochs': config.dnn_epochs,
            'dnn_learning_rate': config.dnn_learning_rate,
            'dnn_layer_dim': config.dnn_layer_dim,
            'dnn_node_dim': config.dnn_node_dim,
            'early_stop_patience': config.early_stop_patience,
            'early_stop_start_epoch': config.early_stop_start_epoch,
            'training_seed': config.training_seed
        },
        'physics_config': {
            'boundary_decay_lambda': config.boundary_decay_lambda,
            'smoothness_decay_lambda': config.smoothness_decay_lambda
        },
        'metrics': metrics
    }
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


# ==============================================================================
# Single Experiment Runner
# ==============================================================================

class SingleExperimentRunner:
    """
    Runner for a single sampling experiment.
    
    Handles complete workflow for one sampling configuration:
    - Internal DNN training with early stopping
    - Evaluation on fixed global validation and test sets
    - Physics consistency evaluation
    - Results saving
    
    Attributes:
        config: Experiment configuration.
        output_dir: Output directory for this experiment.
        logger: Logger instance.
        results: Dictionary storing experiment results.
        input_boundary_model: Input boundary model for physics evaluation.
        output_boundary_model: Output boundary model for physics evaluation.
    """
    
    def __init__(
        self,
        config: SmallSampleConfig,
        output_dir: Path,
        input_boundary_model: Optional[BinaryPredictor] = None,
        output_boundary_model: Optional[BinaryPredictor] = None
    ) -> None:
        """
        Initialize single experiment runner.
        
        Args:
            config: Experiment configuration.
            output_dir: Output directory for this experiment.
            input_boundary_model: Pre-loaded input boundary model.
            output_boundary_model: Pre-loaded output boundary model.
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.results: Dict[str, Any] = {}
        self.input_boundary_model = input_boundary_model
        self.output_boundary_model = output_boundary_model
    
    def train_dnn_with_internal_split(
        self,
        X_internal_train: np.ndarray,
        y_internal_train: np.ndarray,
        X_internal_val: np.ndarray,
        y_internal_val: np.ndarray
    ) -> Tuple[Dict[str, Any], nn.Module, StandardScaler, StandardScaler, Dict[str, List]]:
        """
        Train DNN using internal train/val split with early stopping.
        
        Args:
            X_internal_train: Internal training features.
            y_internal_train: Internal training targets.
            X_internal_val: Internal validation features.
            y_internal_val: Internal validation targets.
        
        Returns:
            Tuple of:
                - result: Dictionary with metrics and predictions.
                - dnn: Trained model.
                - xScaler: Feature scaler.
                - yScaler: Target scaler.
                - history: Training history.
        """
        device = torch.device(self.config.device)
        
        # Data standardization
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        
        X_train_scaled = xScaler.fit_transform(X_internal_train)
        y_train_scaled = yScaler.fit_transform(y_internal_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = xScaler.transform(X_internal_val)
        y_val_scaled = yScaler.transform(y_internal_val.reshape(-1, 1)).flatten()
        
        # Set random seed for model initialization
        torch.manual_seed(self.config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training_seed)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
        
        # Create model
        input_dim = X_internal_train.shape[1]
        dnn = DNN(
            in_dim=input_dim,
            out_dim=1,
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim
        ).to(device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(dnn.parameters(), lr=self.config.dnn_learning_rate)
        criterion = nn.MSELoss()
        
        # Early stopping parameters
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training history
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }
        
        # Training loop
        for epoch in range(self.config.dnn_epochs):
            # Training mode
            dnn.train()
            optimizer.zero_grad()
            y_pred = dnn(X_train_tensor)
            train_loss = criterion(y_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Validation mode
            dnn.eval()
            with torch.no_grad():
                y_val_pred = dnn(X_val_tensor)
                val_loss = criterion(y_val_pred, y_val_tensor)
                
                # Calculate R² for monitoring
                y_train_pred_np = dnn(X_train_tensor).cpu().numpy()
                y_val_pred_np = y_val_pred.cpu().numpy()
                
                y_train_true_np = y_train_tensor.cpu().numpy()
                y_val_true_np = y_val_tensor.cpu().numpy()
                
                train_r2 = 1 - np.sum((y_train_true_np - y_train_pred_np)**2) / \
                           np.sum((y_train_true_np - y_train_true_np.mean())**2)
                val_r2 = 1 - np.sum((y_val_true_np - y_val_pred_np)**2) / \
                         np.sum((y_val_true_np - y_val_true_np.mean())**2)
            
            # Record history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            
            # Early stopping
            if val_loss.item() < best_val_loss and epoch >= self.config.early_stop_start_epoch:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(dnn.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stop_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            dnn.load_state_dict(best_model_state)
        
        # Calculate final internal metrics
        dnn.eval()
        with torch.no_grad():
            y_train_pred_scaled = dnn(X_train_tensor).cpu().numpy()
            y_val_pred_scaled = dnn(X_val_tensor).cpu().numpy()
        
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        
        internal_metrics = {
            'train_r2': metrics.r2_score(y_internal_train, y_train_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_internal_train, y_train_pred)),
            'train_mae': metrics.mean_absolute_error(y_internal_train, y_train_pred),
            'val_r2': metrics.r2_score(y_internal_val, y_val_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_internal_val, y_val_pred)),
            'val_mae': metrics.mean_absolute_error(y_internal_val, y_val_pred)
        }
        
        result = {
            'metrics': internal_metrics,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            },
            'true_values': {
                'train': y_internal_train,
                'val': y_internal_val
            }
        }
        
        return result, dnn, xScaler, yScaler, history
    
    def evaluate_with_global_data(
        self,
        dnn: nn.Module,
        xScaler: StandardScaler,
        yScaler: StandardScaler,
        X_sampled_full: np.ndarray,
        y_sampled_full: np.ndarray,
        X_global_val: np.ndarray,
        y_global_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on fixed global validation and test sets.
        
        Args:
            dnn: Trained model.
            xScaler: Feature scaler.
            yScaler: Target scaler.
            X_sampled_full: Full sampled training features.
            y_sampled_full: Full sampled training targets.
            X_global_val: Global validation features.
            y_global_val: Global validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary with global metrics, predictions, and true values.
        """
        device = torch.device(self.config.device)
        dnn.eval()
        
        # Evaluate on full sampled training set
        X_train_scaled = xScaler.transform(X_sampled_full)
        with torch.no_grad():
            y_train_pred_scaled = dnn(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        
        # Evaluate on global validation set
        X_val_scaled = xScaler.transform(X_global_val)
        with torch.no_grad():
            y_val_pred_scaled = dnn(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        
        # Evaluate on test set
        X_test_scaled = xScaler.transform(X_test)
        with torch.no_grad():
            y_test_pred_scaled = dnn(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        y_test_pred = yScaler.inverse_transform(y_test_pred_scaled).flatten()
        
        # Calculate global metrics
        global_metrics = {
            'train_r2': metrics.r2_score(y_sampled_full, y_train_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_sampled_full, y_train_pred)),
            'train_mae': metrics.mean_absolute_error(y_sampled_full, y_train_pred),
            'val_r2': metrics.r2_score(y_global_val, y_val_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_global_val, y_val_pred)),
            'val_mae': metrics.mean_absolute_error(y_global_val, y_val_pred),
            'test_r2': metrics.r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'test_mae': metrics.mean_absolute_error(y_test, y_test_pred)
        }
        
        result = {
            'metrics': global_metrics,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred
            },
            'true_values': {
                'train': y_sampled_full,
                'val': y_global_val,
                'test': y_test
            }
        }
        
        return result
    
    def save_predictions_to_excel(
        self,
        result_global: Dict[str, Any],
        X_sampled_full: np.ndarray,
        X_global_val: np.ndarray,
        X_test: np.ndarray
    ) -> None:
        """
        Save predictions to Excel files.
        
        Args:
            result_global: Global evaluation results.
            X_sampled_full: Full sampled training features.
            X_global_val: Global validation features.
            X_test: Test features.
        """
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True, parents=True)
        
        # Training set predictions
        df_train = pd.DataFrame({
            'Temperature': X_sampled_full[:, 0],
            'Input_Composition': X_sampled_full[:, 1],
            'Output_True': result_global['true_values']['train'],
            'Output_Pred': result_global['predictions']['train'],
            'Error': result_global['true_values']['train'] - result_global['predictions']['train']
        })
        df_train.to_excel(excel_dir / f'{self.config.excel_prefix}train_predictions.xlsx', index=False)
        
        # Validation set predictions
        df_val = pd.DataFrame({
            'Temperature': X_global_val[:, 0],
            'Input_Composition': X_global_val[:, 1],
            'Output_True': result_global['true_values']['val'],
            'Output_Pred': result_global['predictions']['val'],
            'Error': result_global['true_values']['val'] - result_global['predictions']['val']
        })
        df_val.to_excel(excel_dir / f'{self.config.excel_prefix}val_predictions.xlsx', index=False)
        
        # Test set predictions
        df_test = pd.DataFrame({
            'Temperature': X_test[:, 0],
            'Input_Composition': X_test[:, 1],
            'Output_True': result_global['true_values']['test'],
            'Output_Pred': result_global['predictions']['test'],
            'Error': result_global['true_values']['test'] - result_global['predictions']['test']
        })
        df_test.to_excel(excel_dir / f'{self.config.excel_prefix}test_predictions.xlsx', index=False)
    
    def save_metrics_to_excel(self, result_global: Dict[str, Any]) -> None:
        """
        Save metrics to Excel file using standardized function.
        
        Args:
            result_global: Global evaluation results.
        """
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True, parents=True)
        
        # Get physics results (if available)
        physics_results = self.results.get('physical_evaluation', None)
        
        # Use the standardized save function
        save_fold_metrics_to_excel(
            metrics_dict=result_global['metrics'],
            physics_results=physics_results,
            save_path=excel_dir / f'{self.config.excel_prefix}metrics.xlsx',
            logger_instance=self.logger
        )
    
    def run_physical_evaluation(
        self,
        dnn: nn.Module,
        xScaler: StandardScaler,
        yScaler: StandardScaler
    ) -> None:
        """
        Run physical consistency evaluation.
        
        Args:
            dnn: Trained model.
            xScaler: Feature scaler.
            yScaler: Target scaler.
        """
        if self.input_boundary_model is None or self.output_boundary_model is None:
            self.logger.warning("")
            self.logger.warning("="*70)
            self.logger.warning("Boundary models not available")
            self.logger.warning("Skipping physics evaluation")
            self.logger.warning("="*70)
            return
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Physical Consistency Evaluation")
        self.logger.info("="*70)
        
        # Generate phase diagram prediction data
        T_range = (-40, 200)
        W_input_range = (0, 55)
        resolution = 100
        
        self.logger.info(f"Generating phase diagram predictions...")
        self.logger.info(f"  Temperature range: {T_range[0]}°C to {T_range[1]}°C")
        self.logger.info(f"  Composition range: {W_input_range[0]}% to {W_input_range[1]}%")
        self.logger.info(f"  Resolution: {resolution}x{resolution} = {resolution*resolution} points")
        
        T_test = np.linspace(T_range[0], T_range[1], resolution)
        W_input_test = np.linspace(W_input_range[0], W_input_range[1], resolution)
        T_mesh, W_input_mesh = np.meshgrid(T_test, W_input_test)
        X_phase = np.column_stack([T_mesh.ravel(), W_input_mesh.ravel()])
        
        # Predict
        device = torch.device(self.config.device)
        dnn.eval()
        X_phase_scaled = xScaler.transform(X_phase)
        with torch.no_grad():
            y_phase_pred = dnn(torch.FloatTensor(X_phase_scaled).to(device)).cpu().numpy()
        y_phase_pred = yScaler.inverse_transform(y_phase_pred).flatten()
        
        predicted_data = np.column_stack([X_phase[:, 0], X_phase[:, 1], y_phase_pred])
        
        self.logger.info(f"Creating physics evaluator...")
        self.logger.info(f"  Boundary decay lambda: {self.config.boundary_decay_lambda}")
        self.logger.info(f"  Smoothness decay lambda: {self.config.smoothness_decay_lambda}")
        
        # Create physics evaluator 
        evaluator = PhysicalConsistencyEvaluator(
            input_boundary_model=self.input_boundary_model,
            output_boundary_model=self.output_boundary_model,
            boundary_decay_lambda=self.config.boundary_decay_lambda,
            smoothness_decay_lambda=self.config.smoothness_decay_lambda
        )
        
        # Run evaluation
        self.logger.info(f"Running physics evaluation...")
        physics_score, physics_results = evaluator.evaluate_full(
            dnn, xScaler, yScaler, predicted_data, self.config.device
        )
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"Physics Evaluation Complete")
        self.logger.info("="*70)
        self.logger.info(f"Overall Physics Score: {physics_score:.6f}")
        
        # Extract sub-scores if available
        if 'physical_plausibility' in physics_results:
            phys_plaus = physics_results['physical_plausibility']
            if 'boundary_consistency' in phys_plaus and phys_plaus['boundary_consistency']:
                boundary_score = phys_plaus['boundary_consistency'].get('score', 'N/A')
                self.logger.info(f"  - Boundary Consistency: {boundary_score}")
            if 'thermodynamic_smoothness' in phys_plaus and phys_plaus['thermodynamic_smoothness']:
                smoothness_score = phys_plaus['thermodynamic_smoothness'].get('score', 'N/A')
                self.logger.info(f"  - Thermodynamic Smoothness: {smoothness_score}")
        
        self.logger.info("="*70)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(physics_results)
        report_path = self.output_dir / 'data' / 'physical_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Physics evaluation report saved: {report_path}")
        
        # Store results
        self.results['physical_evaluation'] = physics_results
        self.results['physics_score'] = physics_score
        self.results['predicted_data'] = predicted_data
    
    def run(
        self,
        train_pool: Tuple[np.ndarray, np.ndarray],
        global_val_set: Tuple[np.ndarray, np.ndarray],
        test_set: Tuple[np.ndarray, np.ndarray],
        fraction: float,
        sampling_idx: int
    ) -> Dict[str, Any]:
        """
        Run complete experiment workflow.
        
        Args:
            train_pool: Training pool data.
            global_val_set: Global validation set data.
            test_set: Test set data.
            fraction: Sampling fraction.
            sampling_idx: Sampling index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"Experiment: Fraction={int(fraction*100)}%, Sampling={sampling_idx}")
        self.logger.info("="*70)
        
        # Step 1: Sample and split
        X_pool, y_pool = train_pool
        X_global_val, y_global_val = global_val_set
        X_test, y_test = test_set
        
        (X_sampled_full, y_sampled_full,
         X_internal_train, y_internal_train,
         X_internal_val, y_internal_val,
         sampling_seed) = sample_and_split_internal(
            train_pool, fraction, sampling_idx,
            self.config.internal_val_ratio,
            self.config.sampling_base_seed
        )
        
        self.logger.info(f"Data sampling completed:")
        self.logger.info(f"  Sampled: {len(X_sampled_full)} points")
        self.logger.info(f"  Internal train: {len(X_internal_train)} points")
        self.logger.info(f"  Internal val: {len(X_internal_val)} points")
        self.logger.info(f"  Global val: {len(X_global_val)} points")
        self.logger.info(f"  Test: {len(X_test)} points")
        
        # Step 2: Train DNN
        self.logger.info("Training DNN with internal validation...")
        result_internal, dnn, xScaler, yScaler, history = self.train_dnn_with_internal_split(
            X_internal_train, y_internal_train,
            X_internal_val, y_internal_val
        )
        
        self.logger.info(f"Training completed:")
        self.logger.info(f"  Internal train R²: {result_internal['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Internal val R²: {result_internal['metrics']['val_r2']:.6f}")
        
        # Step 3: Evaluate on global data
        self.logger.info("Evaluating on global validation and test sets...")
        result_global = self.evaluate_with_global_data(
            dnn, xScaler, yScaler,
            X_sampled_full, y_sampled_full,
            X_global_val, y_global_val,
            X_test, y_test
        )
        
        self.logger.info(f"Global evaluation completed:")
        self.logger.info(f"  Train R² (full sampled): {result_global['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (global): {result_global['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R²: {result_global['metrics']['test_r2']:.6f}")
        
        # Step 4: Physics evaluation
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            self.logger.info("")
            self.logger.info("Step 4: Physics Evaluation")
            self.run_physical_evaluation(dnn, xScaler, yScaler)
        else:
            self.logger.warning("")
            self.logger.warning("="*70)
            self.logger.warning("Step 4: Physics Evaluation SKIPPED")
            self.logger.warning("Boundary models were not loaded successfully")
            self.logger.warning("="*70)
        
        # Step 5: Save results
        if self.config.save_predictions:
            self.logger.info("Saving predictions...")
            self.save_predictions_to_excel(result_global, X_sampled_full, X_global_val, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result_global)
        
        # Save configuration
        sampling_info = {
            'fraction': fraction,
            'sampling_idx': sampling_idx,
            'sampling_seed': sampling_seed,
            'n_sampled_full': len(X_sampled_full),
            'n_internal_train': len(X_internal_train),
            'n_internal_val': len(X_internal_val),
            'n_global_val': len(X_global_val),
            'n_test': len(X_test)
        }
        save_config(result_global['metrics'], self.config, self.output_dir, sampling_info)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        # Prepare return dict
        return_dict = {
            'fraction': fraction,
            'sampling_idx': sampling_idx,
            'sampling_seed': sampling_seed,
            'n_sampled_full': len(X_sampled_full),
            'n_internal_train': len(X_internal_train),
            'n_internal_val': len(X_internal_val),
            'metrics': result_global['metrics'],
            'physics_score': self.results.get('physics_score'),
            'physics_boundary': None,
            'physics_smoothness': None,
            'elapsed_time': elapsed_time
        }
        
        # Extract physics sub-scores if available
        if self.results.get('physical_evaluation'):
            phys_eval = self.results['physical_evaluation']
            if 'physical_plausibility' in phys_eval:
                phys_plaus = phys_eval['physical_plausibility']
                if 'boundary_consistency' in phys_plaus and phys_plaus['boundary_consistency']:
                    return_dict['physics_boundary'] = phys_plaus['boundary_consistency'].get('score')
                if 'thermodynamic_smoothness' in phys_plaus and phys_plaus['thermodynamic_smoothness']:
                    return_dict['physics_smoothness'] = phys_plaus['thermodynamic_smoothness'].get('score')
        
        return return_dict


# ==============================================================================
# Experiment Manager
# ==============================================================================

class SmallSampleExperimentManager:
    """
    Manager for complete small sample sensitivity study.
    
    Coordinates multiple sampling experiments across different fractions:
    - Loads fixed train/val/test sets
    - Loads boundary models for physics evaluation
    - Runs experiments for each fraction with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_pool: Training pool data.
        global_val_set: Global validation set data.
        test_set: Test set data.
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: SmallSampleConfig) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        # Load data
        self.logger.info("Loading datasets...")
        self.train_pool = self._load_train_pool()
        self.global_val_set = self._load_global_val_set()
        self.test_set = self._load_test_set()
        
        self.logger.info(f"Data loaded:")
        self.logger.info(f"  Training pool: {len(self.train_pool[0])} samples")
        self.logger.info(f"  Global validation: {len(self.global_val_set[0])} samples")
        self.logger.info(f"  Test: {len(self.test_set[0])} samples")
        
        # Load boundary models
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        # Results storage
        self.all_results: Dict[float, List[Dict[str, Any]]] = {
            fraction: [] for fraction in config.sample_fractions
        }
    
    def _load_train_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training pool data."""
        train_path = self.config.data_dir / self.config.train_pool_file
        return load_ternary_data(train_path)
    
    def _load_global_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load global validation set data."""
        val_path = self.config.data_dir / self.config.validation_file
        return load_ternary_data(val_path)
    
    def _load_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test set data."""
        test_path = self.config.data_dir / self.config.test_file
        return load_ternary_data(test_path)
    
    def _load_boundary_models(self) -> None:
        """Load boundary models for physics evaluation."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Loading Boundary Models for Physics Evaluation")
        self.logger.info("="*70)
        
        input_model_path = self.config.models_dir / self.config.input_boundary_model_file
        output_model_path = self.config.models_dir / self.config.output_boundary_model_file
        
        self.logger.info(f"Models directory: {self.config.models_dir}")
        self.logger.info(f"Input model file: {self.config.input_boundary_model_file}")
        self.logger.info(f"Output model file: {self.config.output_boundary_model_file}")
        self.logger.info(f"Input model path exists: {input_model_path.exists()}")
        self.logger.info(f"Output model path exists: {output_model_path.exists()}")
        
        try:
            if not input_model_path.exists():
                raise FileNotFoundError(f"Input boundary model not found: {input_model_path}")
            
            if not output_model_path.exists():
                raise FileNotFoundError(f"Output boundary model not found: {output_model_path}")
            
            self.logger.info(f"Loading input boundary model from: {input_model_path}")
            self.input_boundary_model = BinaryPredictor.load(str(input_model_path))
            self.logger.info("Input boundary model loaded successfully")
            
            self.logger.info(f"Loading output boundary model from: {output_model_path}")
            self.output_boundary_model = BinaryPredictor.load(str(output_model_path))
            self.logger.info("Output boundary model loaded successfully")
            
            self.logger.info("")
            self.logger.info("All boundary models loaded - Physics evaluation ENABLED")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error("")
            self.logger.error("="*70)
            self.logger.error("FAILED to load boundary models")
            self.logger.error("="*70)
            self.logger.error(f"Error: {e}")
            self.logger.error("")
            self.logger.error("Physics evaluation will be SKIPPED")
            self.logger.error("")
            self.logger.error("To enable physics evaluation, ensure boundary models exist at:")
            self.logger.error(f"  1. {input_model_path}")
            self.logger.error(f"  2. {output_model_path}")
            self.logger.error("="*70)
            
            self.input_boundary_model = None
            self.output_boundary_model = None
    
    def run_all_experiments(self) -> None:
        """Run all sampling experiments."""
        total_experiments = len(self.config.sample_fractions) * self.config.n_sampling_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"Starting Small Sample Sensitivity Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}")
        self.logger.info(f"Repeats per fraction: {self.config.n_sampling_repeats}")
        
        experiment_count = 0
        
        # Run experiments for each fraction
        for fraction in self.config.sample_fractions:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Running experiments for {int(fraction*100)}% sampling")
            self.logger.info("="*70)
            
            for sampling_idx in range(self.config.n_sampling_repeats):
                experiment_count += 1
                
                # Create output directory
                output_dir = (
                    self.config.output_dir / 
                    f'fraction_{int(fraction*100):03d}' /
                    f'sampling_{sampling_idx:02d}'
                )
                
                # Run single experiment
                runner = SingleExperimentRunner(
                    self.config,
                    output_dir,
                    self.input_boundary_model,
                    self.output_boundary_model
                )
                
                result = runner.run(
                    self.train_pool,
                    self.global_val_set,
                    self.test_set,
                    fraction,
                    sampling_idx
                )
                
                self.all_results[fraction].append(result)
                
                self.logger.info(f"Progress: {experiment_count}/{total_experiments} completed")
        
        # Save aggregated results
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Saving aggregated results")
        self.logger.info("="*70)
        
        self._save_aggregated_results()
        self._generate_summary_report()
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Small Sample Sensitivity Study completed successfully")
        self.logger.info("="*70)
    
    def _save_aggregated_results(self) -> None:
        """Save aggregated results across all experiments."""
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare data for summary
        data = []
        
        for fraction in self.config.sample_fractions:
            results = self.all_results[fraction]
            
            if len(results) == 0:
                continue
            
            row = {
                'Sample_Fraction': f'{int(fraction*100)}%',
                'n_sampled_full': results[0]['n_sampled_full'],
                'n_internal_train': results[0]['n_internal_train'],
                'n_internal_val': results[0]['n_internal_val'],
                'n_samplings': len(results)
            }
            
            # Validation R²
            val_r2_values = [r['metrics']['val_r2'] for r in results]
            row['Val_R2_Mean'] = np.mean(val_r2_values)
            row['Val_R2_Std'] = np.std(val_r2_values, ddof=1)
            
            # Test R²
            test_r2_values = [r['metrics']['test_r2'] for r in results]
            row['Test_R2_Mean'] = np.mean(test_r2_values)
            row['Test_R2_Std'] = np.std(test_r2_values, ddof=1)
            
            # RMSE
            rmse_values = [r['metrics']['test_rmse'] for r in results]
            row['Test_RMSE_Mean'] = np.mean(rmse_values)
            row['Test_RMSE_Std'] = np.std(rmse_values, ddof=1)
            
            # MAE
            mae_values = [r['metrics']['test_mae'] for r in results]
            row['Test_MAE_Mean'] = np.mean(mae_values)
            row['Test_MAE_Std'] = np.std(mae_values, ddof=1)
            
            # Physics score
            phys_values = [r.get('physics_score', float('nan')) for r in results]
            phys_values = [v for v in phys_values if v is not None and not np.isnan(v)]
            if len(phys_values) > 0:
                row['Physics_Total_Mean'] = np.mean(phys_values)
                row['Physics_Total_Std'] = np.std(phys_values, ddof=1) if len(phys_values) > 1 else 0.0
            else:
                row['Physics_Total_Mean'] = np.nan
                row['Physics_Total_Std'] = np.nan
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        metrics_path = summary_dir / 'overall_metrics.xlsx'
        df.to_excel(metrics_path, index=False)
        
        self.logger.info(f"Aggregated metrics saved: {metrics_path.name}")
    
    def _generate_summary_report(self) -> None:
        """Generate comprehensive text report."""
        summary_dir = self.config.output_dir / 'summary'
        
        report = []
        report.append("="*70)
        report.append("Small Sample Sensitivity Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Version: v3.0 (Standardized - Baseline)")
        report.append(f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}")
        report.append(f"Repeats per fraction: {self.config.n_sampling_repeats}")
        report.append(f"Total experiments: {len(self.config.sample_fractions) * self.config.n_sampling_repeats}")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Internal split ratio: {100-self.config.internal_val_ratio*100:.0f}% train / {self.config.internal_val_ratio*100:.0f}% val")
        report.append(f"Early stopping: Internal validation set")
        report.append(f"Independent evaluation: Fixed global validation and test sets")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training pool: {len(self.train_pool[0])} samples")
        report.append(f"Global validation: {len(self.global_val_set[0])} samples")
        report.append(f"Test set: {len(self.test_set[0])} samples")
        
        report.append("\n" + "-"*70)
        report.append("Results Summary (Test R² - Independent Evaluation)")
        report.append("-"*70)
        
        for fraction in self.config.sample_fractions:
            results = self.all_results[fraction]
            if len(results) == 0:
                continue
            
            r2_values = [r['metrics']['test_r2'] for r in results]
            
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values, ddof=1)
            
            n_full = results[0]['n_sampled_full']
            n_train = results[0]['n_internal_train']
            n_val = results[0]['n_internal_val']
            
            report.append(
                f"{int(fraction*100):3d}% ({n_full:3d} full = {n_train:3d} train + {n_val:2d} val): "
                f"R² = {mean_r2:.6f} ± {std_r2:.6f}"
            )
        
        report.append("\n" + "="*70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = summary_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Summary report saved: {report_path.name}")
        self.logger.info("\n" + report_text)


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """
    Main entry point for small sample sensitivity experiment.
    
    Configures and runs complete small sample sensitivity study.
    """
    # Create configuration
    config = SmallSampleConfig()
    
    # Configure experiment
    config.sample_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    config.n_sampling_repeats = 10
    config.internal_val_ratio = 0.2
    config.training_seed = 42
    config.sampling_base_seed = 10000
    
    config.dnn_epochs = 1000
    config.dnn_learning_rate = 0.008311634679852062
    config.dnn_layer_dim = 4
    config.dnn_node_dim = 128
    config.early_stop_patience = 1000
    config.early_stop_start_epoch = 500
    
    config.boundary_decay_lambda = 5.0
    config.smoothness_decay_lambda = 15.0
    
    config.save_predictions = True
    config.save_metrics = True
    config.log_level = logging.INFO
    
    # Log project information
    logger.info("")
    logger.info("="*70)
    logger.info("Project Paths")
    logger.info("="*70)
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"  - Train: {config.data_dir / config.train_pool_file}")
    logger.info(f"  - Val: {config.data_dir / config.validation_file}")
    logger.info(f"  - Test: {config.data_dir / config.test_file}")
    logger.info(f"Models directory: {config.models_dir}")
    logger.info(f"  - Input boundary: {config.models_dir / config.input_boundary_model_file}")
    logger.info(f"  - Output boundary: {config.models_dir / config.output_boundary_model_file}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Small Sample Sensitivity Experiment v3.0 (Standardized - Baseline)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Sample fractions: {[f'{int(f*100)}%' for f in config.sample_fractions]}")
    logger.info(f"Repeats per fraction: {config.n_sampling_repeats}")
    logger.info(f"Total experiments: {len(config.sample_fractions) * config.n_sampling_repeats}")
    logger.info("")
    logger.info("Data files:")
    logger.info(f"  Training pool: {config.train_pool_file}")
    logger.info(f"  Global validation: {config.validation_file}")
    logger.info(f"  Test set: {config.test_file}")
    logger.info("")
    logger.info("Key features:")
    logger.info(f"  - Internal split: {100-config.internal_val_ratio*100:.0f}% train / {config.internal_val_ratio*100:.0f}% val")
    logger.info(f"  - Early stopping on internal validation")
    logger.info(f"  - Independent evaluation on fixed global sets")
    logger.info(f"  - Training seed: {config.training_seed} (fixed for reproducibility)")
    logger.info(f"  - Physics evaluation: boundary_lambda={config.boundary_decay_lambda}, smoothness_lambda={config.smoothness_decay_lambda}")
    logger.info("")
    logger.info("Baseline characteristics:")
    logger.info("  - No data augmentation")
    logger.info("  - No outlier detection")
    logger.info("  - No physical constraints")
    
    # Run experiment
    manager = SmallSampleExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("Small sample sensitivity experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()