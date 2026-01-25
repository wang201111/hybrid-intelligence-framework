"""
Noise robustness experiment for solubility system baseline model.

Evaluates baseline DNN robustness to label noise by injecting Gaussian noise
into training data at multiple levels without any data augmentation, outlier
detection, or physical constraints.

Experimental Design:
    1. Load fixed train/val/test sets
    2. For each noise level:
        - Repeat N times:
            - Inject Gaussian noise into training labels
            - Train DNN with fixed training seed
            - Evaluate on clean val/test sets
            - Run physics evaluation
    3. Aggregate results across noise levels
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
# Noise Injection Function
# ==============================================================================

def add_gaussian_noise(
    train_set: Tuple[np.ndarray, np.ndarray],
    noise_level: float,
    noise_idx: int,
    base_seed: int = 20000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Add Gaussian noise to training labels.
    
    Noise is added as a percentage of the label range:
        noise_std = noise_level * (y_max - y_min)
    
    Args:
        train_set: Tuple of (X_train, y_train) clean data.
        noise_level: Noise level as fraction (0.0-0.2).
        noise_idx: Noise repetition index (0-19).
        base_seed: Base random seed for noise generation.
    
    Returns:
        Tuple of:
            - X_train: Features (unchanged).
            - y_noisy: Labels with noise injected.
            - noise_seed: Actual seed used for reproducibility.
    
    Examples:
        >>> X, y_noisy, seed = add_gaussian_noise(
        ...     (X_train, y_train), noise_level=0.10, noise_idx=0
        ... )
    """
    X_train, y_train = train_set
    
    # Generate unique seed for this noise injection
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    
    # No noise case
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed
    
    # Inject Gaussian noise
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    noise_std = noise_level * y_range
    
    noise = np.random.normal(0, noise_std, size=y_train.shape)
    y_noisy = y_train + noise
    
    return X_train.copy(), y_noisy, noise_seed


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class NoiseRobustnessConfig:
    """
    Configuration for noise robustness experiment.
    
    Attributes:
        data_dir: Base data directory.
        train_file: Training data filename.
        validation_file: Validation data filename.
        test_file: Test data filename.
        models_dir: Directory containing boundary models.
        input_boundary_model_file: Input boundary model filename.
        output_boundary_model_file: Output boundary model filename.
        noise_levels: List of noise levels to test (as fractions).
        n_noise_repeats: Number of repetitions per noise level.
        dnn_epochs: Number of training epochs.
        dnn_learning_rate: Learning rate.
        dnn_layer_dim: Number of DNN layers.
        dnn_node_dim: Number of nodes per layer.
        training_seed: Fixed seed for model initialization.
        noise_base_seed: Base seed for noise injection.
        boundary_decay_lambda: Decay coefficient for boundary scoring.
        smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        output_dir: Output directory for results.
        save_predictions: Whether to save predictions to Excel.
        save_metrics: Whether to save metrics to Excel.
        excel_prefix: Prefix for Excel output files.
        device: Computing device ('auto', 'cuda', 'cpu').
        log_level: Logging level.
    """
    # Data paths (absolute paths based on PROJECT_ROOT)
    data_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'fixed_splits'
    train_file: str = 'train.xlsx'
    validation_file: str = 'val.xlsx'
    test_file: str = 'test.xlsx'
    
    # Model paths (absolute paths based on PROJECT_ROOT)
    models_dir: Path = PROJECT_ROOT / 'models' / 'solubility' / 'binary'
    input_boundary_model_file: str = 'KCl_H2O.pth'
    output_boundary_model_file: str = 'MgCl2_H2O.pth'
    
    # Noise configuration
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    n_noise_repeats: int = 10
    
    # Training parameters
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    # Random seeds
    training_seed: int = 42  # Fixed for all experiments
    noise_base_seed: int = 20000  # Base for noise injection
    
    # Physics evaluation parameters
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    # Output settings (absolute path based on PROJECT_ROOT)
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'noise' / 'noise_robustness_baseline_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'noisy_'
    
    # Device
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: auto-detect device."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    config: NoiseRobustnessConfig,
    save_dir: Path,
    noise_info: Dict[str, Any]
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        noise_info: Noise injection information dictionary.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Noise Robustness Study (Baseline)',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device,
            'noise_level': noise_info['noise_level'],
            'noise_index': noise_info['noise_idx'],
            'noise_seed': noise_info['noise_seed']
        },
        'data_info': {
            'n_train': noise_info['n_train'],
            'n_val': noise_info.get('n_val', 'N/A'),
            'n_test': noise_info.get('n_test', 'N/A'),
            'noise_type': 'Gaussian (additive)',
            'noise_target': 'Training labels only'
        },
        'training_config': {
            'dnn_epochs': config.dnn_epochs,
            'dnn_learning_rate': config.dnn_learning_rate,
            'dnn_layer_dim': config.dnn_layer_dim,
            'dnn_node_dim': config.dnn_node_dim,
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

class SingleNoiseExperimentRunner:
    """
    Runner for a single noise experiment.
    
    Handles complete workflow for one noise configuration:
    - Inject Gaussian noise into training labels
    - Train DNN with fixed seed
    - Evaluate on clean validation and test sets
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
        config: NoiseRobustnessConfig,
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
    
    def train_dnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Dict[str, Any], nn.Module, StandardScaler, StandardScaler]:
        """
        Train DNN on noisy training data.
        
        Args:
            X_train: Training features (clean).
            y_train: Training targets (with noise).
            X_val: Validation features (clean).
            y_val: Validation targets (clean).
        
        Returns:
            Tuple of:
                - result: Dictionary with metrics and predictions.
                - dnn: Trained model.
                - xScaler: Feature scaler.
                - yScaler: Target scaler.
        """
        device = torch.device(self.config.device)
        
        # Data standardization
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        
        X_train_scaled = xScaler.fit_transform(X_train)
        y_train_scaled = yScaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = xScaler.transform(X_val)
        y_val_scaled = yScaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Set random seed for model initialization (FIXED across all experiments)
        torch.manual_seed(self.config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training_seed)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
        
        # Create model
        input_dim = X_train.shape[1]
        dnn = DNN(
            in_dim=input_dim,
            out_dim=1,
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim
        ).to(device)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(dnn.parameters(), lr=self.config.dnn_learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(self.config.dnn_epochs):
            # Training mode
            dnn.train()
            optimizer.zero_grad()
            y_pred = dnn(X_train_tensor)
            train_loss = criterion(y_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
        
        # Final evaluation
        dnn.eval()
        with torch.no_grad():
            y_train_pred_scaled = dnn(X_train_tensor).cpu().numpy()
            y_val_pred_scaled = dnn(X_val_tensor).cpu().numpy()
        
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        
        train_metrics = {
            'train_r2': metrics.r2_score(y_train, y_train_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
            'train_mae': metrics.mean_absolute_error(y_train, y_train_pred),
            'val_r2': metrics.r2_score(y_val, y_val_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_val, y_val_pred)),
            'val_mae': metrics.mean_absolute_error(y_val, y_val_pred)
        }
        
        result = {
            'metrics': train_metrics,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred
            },
            'true_values': {
                'train': y_train,
                'val': y_val
            }
        }
        
        return result, dnn, xScaler, yScaler
    
    def evaluate_on_test(
        self,
        dnn: nn.Module,
        xScaler: StandardScaler,
        yScaler: StandardScaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on all datasets.
        
        Args:
            dnn: Trained model.
            xScaler: Feature scaler.
            yScaler: Target scaler.
            X_train: Training features.
            y_train: Training targets (with noise).
            X_val: Validation features.
            y_val: Validation targets (clean).
            X_test: Test features.
            y_test: Test targets (clean).
        
        Returns:
            Dictionary with complete metrics, predictions, and true values.
        """
        device = torch.device(self.config.device)
        dnn.eval()
        
        # Evaluate on training set
        X_train_scaled = xScaler.transform(X_train)
        with torch.no_grad():
            y_train_pred_scaled = dnn(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        
        # Evaluate on validation set
        X_val_scaled = xScaler.transform(X_val)
        with torch.no_grad():
            y_val_pred_scaled = dnn(torch.FloatTensor(X_val_scaled).to(device)).cpu().numpy()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        
        # Evaluate on test set
        X_test_scaled = xScaler.transform(X_test)
        with torch.no_grad():
            y_test_pred_scaled = dnn(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        y_test_pred = yScaler.inverse_transform(y_test_pred_scaled).flatten()
        
        # Calculate complete metrics
        complete_metrics = {
            'train_r2': metrics.r2_score(y_train, y_train_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)),
            'train_mae': metrics.mean_absolute_error(y_train, y_train_pred),
            'val_r2': metrics.r2_score(y_val, y_val_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_val, y_val_pred)),
            'val_mae': metrics.mean_absolute_error(y_val, y_val_pred),
            'test_r2': metrics.r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'test_mae': metrics.mean_absolute_error(y_test, y_test_pred)
        }
        
        result = {
            'metrics': complete_metrics,
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
        
        return result
    
    def save_predictions_to_excel(
        self,
        result: Dict[str, Any],
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> None:
        """
        Save predictions to Excel files.
        
        Args:
            result: Evaluation results.
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.
        """
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True, parents=True)
        
        # Training set predictions
        df_train = pd.DataFrame({
            'Temperature': X_train[:, 0],
            'Input_Composition': X_train[:, 1],
            'Output_True': result['true_values']['train'],
            'Output_Pred': result['predictions']['train'],
            'Error': result['true_values']['train'] - result['predictions']['train']
        })
        df_train.to_excel(excel_dir / f'{self.config.excel_prefix}train_predictions.xlsx', index=False)
        
        # Validation set predictions
        df_val = pd.DataFrame({
            'Temperature': X_val[:, 0],
            'Input_Composition': X_val[:, 1],
            'Output_True': result['true_values']['val'],
            'Output_Pred': result['predictions']['val'],
            'Error': result['true_values']['val'] - result['predictions']['val']
        })
        df_val.to_excel(excel_dir / f'{self.config.excel_prefix}val_predictions.xlsx', index=False)
        
        # Test set predictions
        df_test = pd.DataFrame({
            'Temperature': X_test[:, 0],
            'Input_Composition': X_test[:, 1],
            'Output_True': result['true_values']['test'],
            'Output_Pred': result['predictions']['test'],
            'Error': result['true_values']['test'] - result['predictions']['test']
        })
        df_test.to_excel(excel_dir / f'{self.config.excel_prefix}test_predictions.xlsx', index=False)
    
    def save_metrics_to_excel(self, result: Dict[str, Any]) -> None:
        """
        Save metrics to Excel file using standardized function.
        
        Args:
            result: Evaluation results.
        """
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True, parents=True)
        
        # Get physics results (if available)
        physics_results = self.results.get('physical_evaluation', None)
        
        # Use the standardized save function
        save_fold_metrics_to_excel(
            metrics_dict=result['metrics'],
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
        self.logger.info("Physics Evaluation Complete")
        self.logger.info("="*70)
        self.logger.info(f"Overall Physics Score: {physics_score:.6f}")
        
        # Extract sub-scores if available
        boundary_score = physics_results.get('boundary_score', 'N/A')
        smoothness_score = physics_results.get('smoothness_score', 'N/A')
        if boundary_score != 'N/A':
            self.logger.info(f"  - Boundary Consistency: {boundary_score:.6f}")
        if smoothness_score != 'N/A':
            self.logger.info(f"  - Thermodynamic Smoothness: {smoothness_score:.6f}")
        
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
        train_set: Tuple[np.ndarray, np.ndarray],
        val_set: Tuple[np.ndarray, np.ndarray],
        test_set: Tuple[np.ndarray, np.ndarray],
        noise_level: float,
        noise_idx: int
    ) -> Dict[str, Any]:
        """
        Run complete experiment workflow.
        
        Args:
            train_set: Clean training set data (X, y).
            val_set: Validation set data (X, y).
            test_set: Test set data (X, y).
            noise_level: Noise level to inject (0.0-0.2).
            noise_idx: Noise repetition index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"Experiment: Noise={int(noise_level*100)}%, Repeat={noise_idx}")
        self.logger.info("="*70)
        
        # Step 1: Inject noise
        X_train, y_train_noisy, noise_seed = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed
        )
        
        X_val, y_val = val_set
        X_test, y_test = test_set
        
        self.logger.info(f"Noise injection completed:")
        self.logger.info(f"  Noise level: {noise_level*100:.0f}%")
        self.logger.info(f"  Noise seed: {noise_seed}")
        self.logger.info(f"  Train: {len(X_train)} points (noisy labels)")
        self.logger.info(f"  Val: {len(X_val)} points (clean)")
        self.logger.info(f"  Test: {len(X_test)} points (clean)")
        
        # Step 2: Train DNN
        self.logger.info("Training DNN on noisy data...")
        result_train, dnn, xScaler, yScaler = self.train_dnn(
            X_train, y_train_noisy, X_val, y_val
        )
        
        self.logger.info(f"Training completed:")
        self.logger.info(f"  Train R² (noisy labels): {result_train['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (clean labels): {result_train['metrics']['val_r2']:.6f}")
        
        # Step 3: Evaluate on all sets
        self.logger.info("Evaluating on all datasets...")
        result_complete = self.evaluate_on_test(
            dnn, xScaler, yScaler,
            X_train, y_train_noisy, X_val, y_val, X_test, y_test
        )
        
        self.logger.info(f"Complete evaluation:")
        self.logger.info(f"  Train R²: {result_complete['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R²: {result_complete['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R²: {result_complete['metrics']['test_r2']:.6f}")
        
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
            self.save_predictions_to_excel(result_complete, X_train, X_val, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result_complete)
        
        # Save configuration
        noise_info = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test)
        }
        save_config(result_complete['metrics'], self.config, self.output_dir, noise_info)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        # Prepare return dict
        return_dict = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'n_train': len(X_train),
            'metrics': result_complete['metrics'],
            'physics_score': self.results.get('physics_score'),
            'physics_boundary': None,
            'physics_smoothness': None,
            'elapsed_time': elapsed_time
        }
        
        # Extract physics sub-scores if available
        if self.results.get('physical_evaluation'):
            phys_results = self.results['physical_evaluation']
            return_dict['physics_boundary'] = phys_results.get('boundary_score')
            return_dict['physics_smoothness'] = phys_results.get('smoothness_score')
        
        return return_dict


# ==============================================================================
# Experiment Manager
# ==============================================================================

class NoiseRobustnessExperimentManager:
    """
    Manager for complete noise robustness study.
    
    Coordinates multiple noise experiments across different levels:
    - Loads fixed train/val/test sets
    - Loads boundary models for physics evaluation
    - Runs experiments for each noise level with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_set: Clean training set data.
        val_set: Validation set data.
        test_set: Test set data.
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: NoiseRobustnessConfig) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        # Load data
        self.logger.info("Loading datasets...")
        self.train_set = self._load_train_set()
        self.val_set = self._load_val_set()
        self.test_set = self._load_test_set()
        
        self.logger.info(f"Data loaded:")
        self.logger.info(f"  Training set: {len(self.train_set[0])} samples")
        self.logger.info(f"  Validation set: {len(self.val_set[0])} samples")
        self.logger.info(f"  Test set: {len(self.test_set[0])} samples")
        
        # Load boundary models
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        # Results storage
        self.all_results: Dict[float, List[Dict[str, Any]]] = {
            noise_level: [] for noise_level in config.noise_levels
        }
    
    def _load_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training set data."""
        train_path = self.config.data_dir / self.config.train_file
        return load_ternary_data(train_path)
    
    def _load_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation set data."""
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
            self.input_boundary_model = BinaryPredictor.load(
                str(input_model_path)
            )
            self.logger.info("Input boundary model loaded successfully")
            
            self.logger.info(f"Loading output boundary model from: {output_model_path}")
            self.output_boundary_model = BinaryPredictor.load(
                str(output_model_path)
            )
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
        """Run all noise experiments."""
        total_experiments = len(self.config.noise_levels) * self.config.n_noise_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"Starting Noise Robustness Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        self.logger.info(f"Repeats per level: {self.config.n_noise_repeats}")
        
        experiment_count = 0
        
        # Run experiments for each noise level
        for noise_level in self.config.noise_levels:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Running experiments for {int(noise_level*100)}% noise")
            self.logger.info("="*70)
            
            for noise_idx in range(self.config.n_noise_repeats):
                experiment_count += 1
                
                # Create output directory
                output_dir = (
                    self.config.output_dir / 
                    f'noise_{int(noise_level*100):03d}' /
                    f'repeat_{noise_idx:02d}'
                )
                
                # Run single experiment
                runner = SingleNoiseExperimentRunner(
                    self.config,
                    output_dir,
                    self.input_boundary_model,
                    self.output_boundary_model
                )
                
                result = runner.run(
                    self.train_set,
                    self.val_set,
                    self.test_set,
                    noise_level,
                    noise_idx
                )
                
                self.all_results[noise_level].append(result)
                
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
        self.logger.info("Noise Robustness Study completed successfully")
        self.logger.info("="*70)
    
    def _save_aggregated_results(self) -> None:
        """Save aggregated results across all experiments."""
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare data for summary
        data = []
        
        for noise_level in self.config.noise_levels:
            results = self.all_results[noise_level]
            
            if len(results) == 0:
                continue
            
            row = {
                'Noise_Level': f'{int(noise_level*100)}%',
                'n_train': results[0]['n_train'],
                'n_repeats': len(results)
            }
            
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
        report.append("Noise Robustness Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        report.append(f"Repeats per level: {self.config.n_noise_repeats}")
        report.append(f"Total experiments: {len(self.config.noise_levels) * self.config.n_noise_repeats}")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Noise type: Gaussian (additive)")
        report.append(f"Noise target: Training labels only")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        report.append(f"Evaluation: Clean validation and test sets")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training set: {len(self.train_set[0])} samples (noise injected)")
        report.append(f"Validation set: {len(self.val_set[0])} samples (clean)")
        report.append(f"Test set: {len(self.test_set[0])} samples (clean)")
        
        report.append("\n" + "-"*70)
        report.append("Results Summary (Test R²)")
        report.append("-"*70)
        
        for noise_level in self.config.noise_levels:
            results = self.all_results[noise_level]
            if len(results) == 0:
                continue
            
            r2_values = [r['metrics']['test_r2'] for r in results]
            
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values, ddof=1)
            
            report.append(
                f"{int(noise_level*100):3d}% noise: "
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
    Main entry point for noise robustness experiment.
    
    Configures and runs complete noise robustness study.
    """
    # Create configuration
    config = NoiseRobustnessConfig()
    
    # Configure experiment
    config.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    config.n_noise_repeats = 10
    config.training_seed = 42
    config.noise_base_seed = 20000
    
    config.dnn_epochs = 1000
    config.dnn_learning_rate = 0.00831
    config.dnn_layer_dim = 4
    config.dnn_node_dim = 128
    
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
    logger.info(f"  - Train: {config.data_dir / config.train_file}")
    logger.info(f"  - Val: {config.data_dir / config.validation_file}")
    logger.info(f"  - Test: {config.data_dir / config.test_file}")
    logger.info(f"Models directory: {config.models_dir}")
    logger.info(f"  - Input boundary: {config.models_dir / config.input_boundary_model_file}")
    logger.info(f"  - Output boundary: {config.models_dir / config.output_boundary_model_file}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Noise Robustness Experiment - Baseline")
    logger.info("="*70)
    
    # Run experiment
    manager = NoiseRobustnessExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("Noise robustness experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()