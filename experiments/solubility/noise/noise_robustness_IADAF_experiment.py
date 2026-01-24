"""
================================================================================
IADAF Noise Robustness Experiment (Solubility System)
================================================================================

Tests IADAF's robustness under different noise levels to evaluate its ability
to handle noisy training data.

Key Features:
    - Gaussian noise injection at multiple levels: 0%, 5%, 10%, 15%, 20%
    - Repeated experiments (10-20 repeats per noise level)
    - Internal train/validation split after noise injection
    - IADAF trained on noisy internal data
    - TSTR evaluation framework (Train on Synthetic, Test on Real)
    - Independent evaluation on fixed global validation and test sets
    - Complete metrics tracking and physics evaluation

Experimental Design:
    1. Inject Gaussian noise into training labels at specified level
    2. Internal split: 80% internal train, 20% internal validation
    3. Train IADAF on noisy internal train with validation
    4. Generate synthetic data
    5. Train DNN on synthetic data using TSTR
    6. Evaluate on fixed global validation and test sets
    7. Repeat 10-20 times with different noise seeds per level
    
Standardization:
    - PEP 8 compliant
    - Complete type hints (PEP 484)
    - Google-style docstrings (PEP 257)
    - Logging module instead of print
    - English documentation throughout
    - Robust path handling (works from any directory)
    - No hardcoded paths
    - Module names preserved for paper correspondence
    
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

# Need 3 parents to reach project root
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
    PhysicsConfig,
)
from binary_predictor import BinaryPredictor
from iadaf import IADAFTrainer, IADAFConfig

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
    
    # Physics evaluation scores (if available) - flat structure
    if physics_results is not None:
        physics_total = physics_results.get('overall_score', float('nan'))
        data.append(['Physics Score', physics_total])
        
        physics_boundary = physics_results.get('boundary_score', float('nan'))
        data.append(['Boundary Consistency', physics_boundary])
        
        physics_smoothness = physics_results.get('smoothness_score', float('nan'))
        data.append(['Thermodynamic Smoothness', physics_smoothness])
        
        logger_instance.debug(
            f"Physics metrics included: overall={physics_total:.4f}, "
            f"boundary={physics_boundary:.4f}, smoothness={physics_smoothness:.4f}"
        )
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
        raise ValueError(
            f"Expected at least 2 composition columns, found {len(comp_cols)}"
        )
    
    # Solubility: 2 inputs, 1 output
    X = data[[temp_col, comp_cols[0]]].values.astype(np.float32)
    y = data[comp_cols[1]].values.astype(np.float32)
    
    return X, y


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class IADAFNoiseRobustnessConfig:
    """
    Configuration for IADAF noise robustness experiment.
    
    All paths are ABSOLUTE paths based on PROJECT_ROOT for reliability.
    
    Attributes:
        Data Paths (absolute paths):
            data_dir: Base data directory.
            train_file: Training data filename.
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
        
        Noise Configuration:
            noise_levels: List of noise levels to test (0.0-0.2).
            n_noise_repeats: Number of repeats per noise level.
            noise_base_seed: Base seed for noise generation.
        
        Internal Split Configuration:
            internal_val_ratio: Ratio for internal validation split.
            internal_split_seed: Seed for internal split.
        
        Training Parameters:
            dnn_epochs: Number of DNN training epochs (TSTR).
            dnn_learning_rate: DNN learning rate.
            dnn_layer_dim: Number of DNN layers.
            dnn_node_dim: Number of nodes per layer.
        
        Random Seeds:
            training_seed: Fixed seed for model initialization.
        
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
    train_file: str = 'train.xlsx'
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
    
    # Noise configuration
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    n_noise_repeats: int = 10
    noise_base_seed: int = 20000
    
    # Internal split configuration
    internal_val_ratio: float = 0.2
    internal_split_seed: int = 999
    
    # Training parameters
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    # Random seeds
    training_seed: int = 42
    
    # Physics evaluation parameters
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    # Output settings (absolute path based on PROJECT_ROOT)
    output_dir: Path = (
        PROJECT_ROOT / 'results' / 'solubility' / 'noise' / 'noise_robustness_IADAF_results'
    )
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'iadaf_'
    
    # Device
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: auto-detect device."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    
    Args:
        train_set: Tuple of (X, y) original clean data.
        noise_level: Noise level (0.0-0.2 as fraction of range).
        noise_idx: Noise repeat index (0-19).
        base_seed: Base random seed for noise generation.
    
    Returns:
        Tuple of:
            - X_train: Features (unchanged).
            - y_noisy: Labels with injected noise.
            - noise_seed: Actual seed used for noise generation.
    
    Examples:
        >>> X_noisy, y_noisy, seed = add_gaussian_noise(
        ...     train_set=(X_train, y_train),
        ...     noise_level=0.10,
        ...     noise_idx=0
        ... )
    """
    X_train, y_train = train_set
    
    # Generate unique noise seed
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    
    # No noise for level 0
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed
    
    # Calculate noise standard deviation based on output range
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    noise_std = noise_level * y_range
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, size=len(y_train))
    y_noisy = y_train + noise
    
    return X_train.copy(), y_noisy, noise_seed


# ==============================================================================
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    config: IADAFNoiseRobustnessConfig,
    save_dir: Path,
    noise_info: Dict[str, Any]
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        noise_info: Noise information dictionary.
    """
    # Get default IADAF config for documentation
    default_iadaf = IADAFConfig()
    
    config_dict = {
        'experiment_info': {
            'experiment_name': 'IADAF Noise Robustness Study',
            'version': 'v3.0 (Standardized)',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device,
            'noise_level': noise_info['noise_level'],
            'noise_index': noise_info['noise_idx'],
            'noise_seed': noise_info['noise_seed']
        },
        'data_info': {
            'n_train_clean': noise_info['n_train_clean'],
            'n_train_noisy': noise_info['n_train_noisy'],
            'n_internal_train': noise_info['n_internal_train'],
            'n_internal_val': noise_info['n_internal_val'],
            'n_global_val': noise_info.get('n_global_val', 'N/A'),
            'n_test': noise_info.get('n_test', 'N/A'),
            'internal_val_ratio': config.internal_val_ratio,
            'internal_split_seed': config.internal_split_seed
        },
        'training_config': {
            'dnn_epochs': config.dnn_epochs,
            'dnn_learning_rate': config.dnn_learning_rate,
            'dnn_layer_dim': config.dnn_layer_dim,
            'dnn_node_dim': config.dnn_node_dim,
            'training_seed': config.training_seed
        },
        'iadaf_config': {
            'note': 'All parameters use IADAFConfig defaults',
            'system_type': default_iadaf.SYSTEM_TYPE,
            'bo_iterations': default_iadaf.BO_N_ITERATIONS,
            'bo_init_points': default_iadaf.BO_INIT_POINTS,
            'bo_wgan_epochs': default_iadaf.BO_WGAN_EPOCHS,
            'bo_dnn_epochs': default_iadaf.BO_DNN_EPOCHS,
            'wgan_epochs': default_iadaf.WGAN_EPOCHS,
            'batch_size': default_iadaf.BATCH_SIZE,
            'learning_rate_g': default_iadaf.LEARNING_RATE_G,
            'learning_rate_c': default_iadaf.LEARNING_RATE_C,
            'n_samples_to_generate': default_iadaf.N_SAMPLES_TO_GENERATE,
            'xgb_error_threshold': default_iadaf.XGB_ERROR_THRESHOLD
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

class SingleIADAFNoiseExperimentRunner:
    """
    Runner for a single IADAF noise robustness experiment.
    
    Handles complete workflow for one noise configuration:
    - Noise injection into training labels
    - Internal train/val split
    - IADAF training and synthetic data generation
    - DNN training on synthetic data using TSTR
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
        config: IADAFNoiseRobustnessConfig,
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
        
        self.device = torch.device(config.device)
    
    def internal_train_val_split(
        self,
        X_train_noisy: np.ndarray,
        y_train_noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split noisy training data into internal train/val sets.
        
        Args:
            X_train_noisy: Noisy training features.
            y_train_noisy: Noisy training targets.
        
        Returns:
            Tuple of:
                - X_internal_train: Internal training features.
                - y_internal_train: Internal training targets.
                - X_internal_val: Internal validation features.
                - y_internal_val: Internal validation targets.
        """
        self.logger.info("Splitting noisy data into internal train/val sets...")
        self.logger.info(f"  Total noisy samples: {len(X_train_noisy)}")
        self.logger.info(f"  Internal val ratio: {self.config.internal_val_ratio:.1%}")
        self.logger.info(f"  Split seed: {self.config.internal_split_seed}")
        
        X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
            X_train_noisy,
            y_train_noisy,
            test_size=self.config.internal_val_ratio,
            random_state=self.config.internal_split_seed,
            shuffle=True
        )
        
        self.logger.info(f"  Internal train: {len(X_internal_train)} samples")
        self.logger.info(f"  Internal val: {len(X_internal_val)} samples")
        
        return X_internal_train, y_internal_train, X_internal_val, y_internal_val
    
    def train_and_generate_iadaf(
        self,
        X_internal_train: np.ndarray,
        y_internal_train: np.ndarray,
        X_internal_val: np.ndarray,
        y_internal_val: np.ndarray
    ) -> Tuple[np.ndarray, IADAFTrainer]:
        """
        Train IADAF model and generate synthetic data.
        
        Args:
            X_internal_train: Internal training features.
            y_internal_train: Internal training targets.
            X_internal_val: Internal validation features.
            y_internal_val: Internal validation targets.
        
        Returns:
            Tuple of:
                - X_gen_full: Generated synthetic data [N, 3] (T, W_input, W_output).
                - iadaf_trainer: Trained IADAF trainer instance.
        """
        self.logger.info("Training IADAF and generating synthetic data...")
        
        # Combine features and targets for IADAF (expects 3D input)
        X_train_full = np.column_stack([X_internal_train, y_internal_train])
        X_val_full = np.column_stack([X_internal_val, y_internal_val])
        
        # Use IADAFConfig defaults
        iadaf_config = IADAFConfig(
            SYSTEM_TYPE='solubility',
            VERBOSE=False
        )
        
        self.logger.info("  Initializing IADAF trainer...")
        iadaf_trainer = IADAFTrainer(iadaf_config)
        
        self.logger.info("  Starting IADAF training with validation...")
        self.logger.info(f"    Internal train: {len(X_internal_train)} samples")
        self.logger.info(f"    Internal val: {len(X_internal_val)} samples")
        iadaf_trainer.fit_with_validation(X_train_full, X_val_full)
        
        self.logger.info("  Generating synthetic data...")
        X_gen_full = iadaf_trainer.generate()
        
        self.logger.info(f"  Generated {len(X_gen_full)} synthetic samples")
        
        return X_gen_full, iadaf_trainer
    
    def train_dnn_on_synthetic(
        self,
        X_gen: np.ndarray,
        X_internal_val: np.ndarray,
        y_internal_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[nn.Module, StandardScaler, StandardScaler, Dict]:
        """
        Train DNN on synthetic data using TSTR framework.
        
        Uses TSTREvaluator to ensure consistency with baseline experiments.
        
        Args:
            X_gen: Generated synthetic data [N, 3] (T, W_input, W_output).
            X_internal_val: Internal validation features (for early stopping).
            y_internal_val: Internal validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Tuple of:
                - dnn: Trained DNN model.
                - scaler_X: Feature scaler.
                - scaler_y: Target scaler.
                - history: Training history dictionary.
        """
        self.logger.info("Training DNN on synthetic data using TSTR...")
        
        # Create PhysicsConfig for TSTR
        phys_config = PhysicsConfig(
            tstr_epochs=self.config.dnn_epochs,
            tstr_lr=self.config.dnn_learning_rate,
            dnn_layer_dim=self.config.dnn_layer_dim,
            dnn_node_dim=self.config.dnn_node_dim,
            tstr_device=self.config.device
        )
        
        # Initialize TSTREvaluator (correct parameter names)
        evaluator = TSTREvaluator(
            X_val=X_internal_val,
            y_val=y_internal_val,
            X_test=X_test,
            y_test=y_test,
            X_train=X_gen[:, :2],
            y_train=X_gen[:, 2],
            config=phys_config
        )
        
        # Train on synthetic data
        self.logger.info(f"  Training on {len(X_gen)} synthetic samples...")
        result = evaluator.evaluate(
            X_syn=X_gen[:, :2],
            y_syn=X_gen[:, 2],
            epochs=self.config.dnn_epochs,
            verbose=False
        )
        
        # Extract model and scalers (correct key names)
        dnn = result['model']
        scaler_X = result['x_scaler']
        scaler_y = result['y_scaler']
        
        # Extract history
        history = {
            'best_epoch': result.get('best_epoch', -1),
            'best_val_r2': result.get('best_val_r2', float('nan'))
        }
        
        return dnn, scaler_X, scaler_y, history
    
    def evaluate_with_global_data(
        self,
        dnn: nn.Module,
        xScaler: StandardScaler,
        yScaler: StandardScaler,
        X_train_noisy: np.ndarray,
        y_train_noisy: np.ndarray,
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
            X_train_noisy: Full noisy training features.
            y_train_noisy: Full noisy training targets.
            X_global_val: Global validation features.
            y_global_val: Global validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary with global metrics, predictions, and true values.
        """
        dnn.eval()
        
        # Evaluate on full noisy training set
        X_train_scaled = xScaler.transform(X_train_noisy)
        with torch.no_grad():
            y_train_pred_scaled = dnn(
                torch.FloatTensor(X_train_scaled).to(self.device)
            ).cpu().numpy()
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        
        # Evaluate on global validation set
        X_val_scaled = xScaler.transform(X_global_val)
        with torch.no_grad():
            y_val_pred_scaled = dnn(
                torch.FloatTensor(X_val_scaled).to(self.device)
            ).cpu().numpy()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        
        # Evaluate on test set
        X_test_scaled = xScaler.transform(X_test)
        with torch.no_grad():
            y_test_pred_scaled = dnn(
                torch.FloatTensor(X_test_scaled).to(self.device)
            ).cpu().numpy()
        y_test_pred = yScaler.inverse_transform(y_test_pred_scaled).flatten()
        
        # Calculate global metrics
        global_metrics = {
            'train_r2': metrics.r2_score(y_train_noisy, y_train_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train_noisy, y_train_pred)),
            'train_mae': metrics.mean_absolute_error(y_train_noisy, y_train_pred),
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
                'train': y_train_noisy,
                'val': y_global_val,
                'test': y_test
            }
        }
        
        return result
    
    def save_predictions_to_excel(
        self,
        result_global: Dict[str, Any],
        X_train_noisy: np.ndarray,
        X_global_val: np.ndarray,
        X_test: np.ndarray
    ) -> None:
        """
        Save predictions to Excel files.
        
        Args:
            result_global: Global evaluation results.
            X_train_noisy: Full noisy training features.
            X_global_val: Global validation features.
            X_test: Test features.
        """
        excel_dir = self.output_dir / 'excel'
        excel_dir.mkdir(exist_ok=True, parents=True)
        
        # Training set predictions
        df_train = pd.DataFrame({
            'Temperature': X_train_noisy[:, 0],
            'Input_Composition': X_train_noisy[:, 1],
            'Output_True': result_global['true_values']['train'],
            'Output_Pred': result_global['predictions']['train'],
            'Error': result_global['true_values']['train'] - result_global['predictions']['train']
        })
        df_train.to_excel(
            excel_dir / f'{self.config.excel_prefix}train_predictions.xlsx',
            index=False
        )
        
        # Validation set predictions
        df_val = pd.DataFrame({
            'Temperature': X_global_val[:, 0],
            'Input_Composition': X_global_val[:, 1],
            'Output_True': result_global['true_values']['val'],
            'Output_Pred': result_global['predictions']['val'],
            'Error': result_global['true_values']['val'] - result_global['predictions']['val']
        })
        df_val.to_excel(
            excel_dir / f'{self.config.excel_prefix}val_predictions.xlsx',
            index=False
        )
        
        # Test set predictions
        df_test = pd.DataFrame({
            'Temperature': X_test[:, 0],
            'Input_Composition': X_test[:, 1],
            'Output_True': result_global['true_values']['test'],
            'Output_Pred': result_global['predictions']['test'],
            'Error': result_global['true_values']['test'] - result_global['predictions']['test']
        })
        df_test.to_excel(
            excel_dir / f'{self.config.excel_prefix}test_predictions.xlsx',
            index=False
        )
    
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
        dnn.eval()
        X_phase_scaled = xScaler.transform(X_phase)
        with torch.no_grad():
            y_phase_pred = dnn(
                torch.FloatTensor(X_phase_scaled).to(self.device)
            ).cpu().numpy()
        y_phase_pred = yScaler.inverse_transform(y_phase_pred).flatten()
        
        predicted_data = np.column_stack([X_phase[:, 0], X_phase[:, 1], y_phase_pred])
        
        self.logger.info(f"Creating physics evaluator...")
        self.logger.info(f"  Boundary decay lambda: {self.config.boundary_decay_lambda}")
        self.logger.info(f"  Smoothness decay lambda: {self.config.smoothness_decay_lambda}")
        
        # Create physics evaluator (correct parameter passing)
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
        self.logger.info(f"  - Boundary Score: {physics_results.get('boundary_score', 'N/A')}")
        self.logger.info(f"  - Smoothness Score: {physics_results.get('smoothness_score', 'N/A')}")
        self.logger.info("="*70)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(physics_results)
        report_path = self.output_dir / 'data' / 'physical_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Physics evaluation report saved: {report_path}")
        
        # Store results (flat structure)
        self.results['physical_evaluation'] = physics_results
        self.results['physics_score'] = physics_score
        self.results['predicted_data'] = predicted_data
    
    def run(
        self,
        train_set: Tuple[np.ndarray, np.ndarray],
        global_val_set: Tuple[np.ndarray, np.ndarray],
        test_set: Tuple[np.ndarray, np.ndarray],
        noise_level: float,
        noise_idx: int
    ) -> Dict[str, Any]:
        """
        Run complete IADAF noise robustness experiment workflow.
        
        Args:
            train_set: Clean training data.
            global_val_set: Global validation set data.
            test_set: Test set data.
            noise_level: Noise level to inject.
            noise_idx: Noise repeat index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"IADAF Noise Experiment: Level={int(noise_level*100)}%, Repeat={noise_idx}")
        self.logger.info("="*70)
        
        # Step 1: Inject noise
        X_train_clean, y_train_clean = train_set
        X_global_val, y_global_val = global_val_set
        X_test, y_test = test_set
        
        self.logger.info("")
        self.logger.info("Step 1: Noise Injection")
        X_train_noisy, y_train_noisy, noise_seed = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed
        )
        
        self.logger.info(f"  Clean train samples: {len(X_train_clean)}")
        self.logger.info(f"  Noise level: {noise_level:.1%}")
        self.logger.info(f"  Noise seed: {noise_seed}")
        
        # Step 2: Internal split
        self.logger.info("")
        self.logger.info("Step 2: Internal Train/Val Split")
        X_internal_train, y_internal_train, X_internal_val, y_internal_val = \
            self.internal_train_val_split(X_train_noisy, y_train_noisy)
        
        # Step 3: Train IADAF and generate synthetic data
        self.logger.info("")
        self.logger.info("Step 3: IADAF Training and Data Generation")
        X_gen, iadaf_trainer = self.train_and_generate_iadaf(
            X_internal_train, y_internal_train,
            X_internal_val, y_internal_val
        )
        
        # Step 4: Train DNN on synthetic data
        self.logger.info("")
        self.logger.info("Step 4: TSTR - Train DNN on Synthetic Data")
        dnn, xScaler, yScaler, history = self.train_dnn_on_synthetic(
            X_gen, X_internal_val, y_internal_val, X_test, y_test
        )
        
        # Step 5: Evaluate on global data
        self.logger.info("")
        self.logger.info("Step 5: Evaluation on Global Data")
        result_global = self.evaluate_with_global_data(
            dnn, xScaler, yScaler,
            X_train_noisy, y_train_noisy,
            X_global_val, y_global_val,
            X_test, y_test
        )
        
        self.logger.info(f"Global evaluation completed:")
        self.logger.info(f"  Train R² (noisy): {result_global['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (global): {result_global['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R²: {result_global['metrics']['test_r2']:.6f}")
        
        # Step 6: Physics evaluation
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            self.logger.info("")
            self.logger.info("Step 6: Physics Evaluation")
            self.run_physical_evaluation(dnn, xScaler, yScaler)
        else:
            self.logger.warning("")
            self.logger.warning("="*70)
            self.logger.warning("Step 6: Physics Evaluation SKIPPED")
            self.logger.warning("Boundary models were not loaded successfully")
            self.logger.warning("="*70)
        
        # Step 7: Save results
        if self.config.save_predictions:
            self.logger.info("")
            self.logger.info("Saving predictions...")
            self.save_predictions_to_excel(result_global, X_train_noisy, X_global_val, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result_global)
        
        # Save configuration
        noise_info = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'n_train_clean': len(X_train_clean),
            'n_train_noisy': len(X_train_noisy),
            'n_internal_train': len(X_internal_train),
            'n_internal_val': len(X_internal_val),
            'n_global_val': len(X_global_val),
            'n_test': len(X_test),
            'n_synthetic': len(X_gen)
        }
        save_config(result_global['metrics'], self.config, self.output_dir, noise_info)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        # Prepare return dict
        return_dict = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'n_train_clean': len(X_train_clean),
            'n_train_noisy': len(X_train_noisy),
            'n_internal_train': len(X_internal_train),
            'n_internal_val': len(X_internal_val),
            'n_synthetic': len(X_gen),
            'metrics': result_global['metrics'],
            'physics_score': self.results.get('physics_score'),
            'physics_boundary': None,
            'physics_smoothness': None,
            'elapsed_time': elapsed_time
        }
        
        # Extract physics sub-scores if available (flat structure)
        if self.results.get('physical_evaluation'):
            phys_eval = self.results['physical_evaluation']
            return_dict['physics_boundary'] = phys_eval.get('boundary_score')
            return_dict['physics_smoothness'] = phys_eval.get('smoothness_score')
        
        return return_dict


# ==============================================================================
# Experiment Manager
# ==============================================================================

class IADAFNoiseRobustnessExperimentManager:
    """
    Manager for complete IADAF noise robustness study.
    
    Coordinates multiple noise experiments across different levels:
    - Loads fixed train/val/test sets
    - Loads boundary models for physics evaluation
    - Runs experiments for each noise level with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_set: Training data.
        global_val_set: Global validation set data.
        test_set: Test set data.
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: IADAFNoiseRobustnessConfig) -> None:
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
        self.global_val_set = self._load_global_val_set()
        self.test_set = self._load_test_set()
        
        self.logger.info(f"Data loaded:")
        self.logger.info(f"  Training set: {len(self.train_set[0])} samples")
        self.logger.info(f"  Global validation: {len(self.global_val_set[0])} samples")
        self.logger.info(f"  Test: {len(self.test_set[0])} samples")
        
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
            # Correct: BinaryPredictor.load() does not take device parameter
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
        """Run all noise robustness experiments."""
        total_experiments = len(self.config.noise_levels) * self.config.n_noise_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"Starting IADAF Noise Robustness Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Noise levels: {[f'{int(nl*100)}%' for nl in self.config.noise_levels]}")
        self.logger.info(f"Repeats per level: {self.config.n_noise_repeats}")
        
        experiment_count = 0
        
        # Run experiments for each noise level
        for noise_level in self.config.noise_levels:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Running experiments for {int(noise_level*100)}% noise level")
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
                runner = SingleIADAFNoiseExperimentRunner(
                    self.config,
                    output_dir,
                    self.input_boundary_model,
                    self.output_boundary_model
                )
                
                result = runner.run(
                    self.train_set,
                    self.global_val_set,
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
        self.logger.info("IADAF Noise Robustness Study completed successfully")
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
                'n_train_clean': results[0]['n_train_clean'],
                'n_train_noisy': results[0]['n_train_noisy'],
                'n_internal_train': results[0]['n_internal_train'],
                'n_internal_val': results[0]['n_internal_val'],
                'n_synthetic': results[0]['n_synthetic'],
                'n_repeats': len(results)
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
        report.append("IADAF Noise Robustness Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Version: v3.0 (Standardized - with Augmentation)")
        report.append(f"Noise levels: {[f'{int(nl*100)}%' for nl in self.config.noise_levels]}")
        report.append(f"Repeats per level: {self.config.n_noise_repeats}")
        report.append(f"Total experiments: {len(self.config.noise_levels) * self.config.n_noise_repeats}")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Noise injection: Gaussian noise to training labels")
        report.append(f"Internal split ratio: {100-self.config.internal_val_ratio*100:.0f}% train / {self.config.internal_val_ratio*100:.0f}% val")
        report.append(f"IADAF training: Noisy internal train + validation")
        report.append(f"DNN training: Synthetic data (TSTR framework)")
        report.append(f"Independent evaluation: Fixed global validation and test sets")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        report.append(f"Noise base seed: {self.config.noise_base_seed}")
        report.append(f"Internal split seed: {self.config.internal_split_seed} (fixed)")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training set: {len(self.train_set[0])} samples")
        report.append(f"Global validation: {len(self.global_val_set[0])} samples")
        report.append(f"Test set: {len(self.test_set[0])} samples")
        
        report.append("\n" + "-"*70)
        report.append("Results Summary (Test R² - Independent Evaluation)")
        report.append("-"*70)
        
        for noise_level in self.config.noise_levels:
            results = self.all_results[noise_level]
            if len(results) == 0:
                continue
            
            r2_values = [r['metrics']['test_r2'] for r in results]
            
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values, ddof=1)
            
            n_noisy = results[0]['n_train_noisy']
            n_train = results[0]['n_internal_train']
            n_val = results[0]['n_internal_val']
            n_syn = results[0]['n_synthetic']
            
            report.append(
                f"{int(noise_level*100):3d}% noise ({n_noisy:3d} noisy = {n_train:3d} train + {n_val:2d} val → {n_syn:4d} synthetic): "
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
    Main entry point for IADAF noise robustness experiment.
    
    Configures and runs complete IADAF noise robustness study.
    """
    # Create configuration
    config = IADAFNoiseRobustnessConfig()
    
    # Configure experiment
    config.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    config.n_noise_repeats = 10
    config.noise_base_seed = 20000
    
    config.internal_val_ratio = 0.2
    config.internal_split_seed = 999
    
    config.dnn_epochs = 1000
    config.dnn_learning_rate = 0.00831
    config.dnn_layer_dim = 4
    config.dnn_node_dim = 128
    
    config.training_seed = 42
    
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
    logger.info("IADAF Noise Robustness Study v3.0 (Standardized)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Noise levels: {[f'{int(nl*100)}%' for nl in config.noise_levels]}")
    logger.info(f"Repeats per level: {config.n_noise_repeats}")
    logger.info(f"Total experiments: {len(config.noise_levels) * config.n_noise_repeats}")
    logger.info("")
    logger.info("Data files:")
    logger.info(f"  Training set: {config.train_file}")
    logger.info(f"  Global validation: {config.validation_file}")
    logger.info(f"  Test set: {config.test_file}")
    logger.info("")
    logger.info("Key features:")
    logger.info(f"  - Gaussian noise injection to training labels")
    logger.info(f"  - Internal split: {100-config.internal_val_ratio*100:.0f}% train / {config.internal_val_ratio*100:.0f}% val")
    logger.info(f"  - IADAF data augmentation with BO")
    logger.info(f"  - TSTR evaluation framework")
    logger.info(f"  - Independent evaluation on fixed global sets")
    logger.info(f"  - Training seed: {config.training_seed} (fixed for reproducibility)")
    logger.info(f"  - Noise base seed: {config.noise_base_seed}")
    logger.info(f"  - Internal split seed: {config.internal_split_seed} (fixed)")
    logger.info(f"  - Physics evaluation: boundary_lambda={config.boundary_decay_lambda}, smoothness_lambda={config.smoothness_decay_lambda}")
    logger.info("")
    logger.info("IADAF characteristics:")
    logger.info("  - Bayesian optimization for hyperparameters")
    logger.info("  - WGAN-GP for data generation")
    logger.info("  - XGBoost filtering for quality control")
    logger.info("  - All IADAF parameters use defaults from IADAFConfig")
    
    # Run experiment
    manager = IADAFNoiseRobustnessExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("IADAF noise robustness experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()