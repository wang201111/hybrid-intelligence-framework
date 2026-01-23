"""
Small sample sensitivity experiment with T-KMeans-LOF outlier detection.

Evaluates DNN performance sensitivity to training set size using repeated
random sampling combined with KMeans-LOF data cleaning. This experiment
assesses the impact of both sample size and data quality on model performance.

Experimental Design:
    1. Load fixed train/val/test sets
    2. For each sample fraction (10%, 25%, 50%, 75%, 100%):
        - Repeat N times:
            - Sample from training pool
            - Apply KMeans-LOF cleaning
            - Split cleaned data into internal train/val
            - Train DNN with early stopping on internal val
            - Evaluate on global val/test sets
            - Run physics evaluation
    3. Aggregate results across sample sizes
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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from utils_solubility import (
    DNN,
    PhysicalConsistencyEvaluator,
)
from binary_predictor import BinaryPredictor
from t_kmeans_lof import TKMeansLOF

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
    
    Creates a two-column Excel file with metrics and their values.
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics.
        physics_results: Optional physics evaluation results.
        save_path: Path to save Excel file.
        logger_instance: Optional logger instance.
    """
    if logger_instance is None:
        logger_instance = logger
    
    data = []
    
    if 'metrics' in metrics_dict:
        inner_metrics = metrics_dict['metrics']
    else:
        inner_metrics = metrics_dict
    
    data.append(['Train RMSE', inner_metrics.get('train_rmse', float('nan'))])
    data.append(['Train MAE', inner_metrics.get('train_mae', float('nan'))])
    data.append(['Train R²', inner_metrics.get('train_r2', float('nan'))])
    
    data.append(['Val RMSE', inner_metrics.get('val_rmse', float('nan'))])
    data.append(['Val MAE', inner_metrics.get('val_mae', float('nan'))])
    data.append(['Val R²', inner_metrics.get('val_r2', float('nan'))])
    
    data.append(['Test RMSE', inner_metrics.get('test_rmse', float('nan'))])
    data.append(['Test MAE', inner_metrics.get('test_mae', float('nan'))])
    data.append(['Test R²', inner_metrics.get('test_r2', float('nan'))])
    
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
    
    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    
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
# Sampling Function
# ==============================================================================

def sample_from_pool(
    train_pool: Tuple[np.ndarray, np.ndarray],
    fraction: float,
    sampling_idx: int,
    base_seed: int = 10000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Sample from training pool with fixed random seed.
    
    Args:
        train_pool: Tuple of (X, y) training pool data.
        fraction: Sampling fraction (0.0-1.0).
        sampling_idx: Sampling repetition index.
        base_seed: Base random seed.
    
    Returns:
        Tuple of (X_sample, y_sample, sampling_seed).
    """
    X_pool, y_pool = train_pool
    
    sampling_seed = base_seed + int(fraction * 1000) + sampling_idx * 100
    
    np.random.seed(sampling_seed)
    n_total = len(X_pool)
    n_sample = int(n_total * fraction)
    
    indices = np.random.choice(n_total, size=n_sample, replace=False)
    
    return X_pool[indices].copy(), y_pool[indices].copy(), sampling_seed


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class KMeansLOFSensitivityConfig:
    """
    Configuration for KMeans-LOF sample sensitivity experiment.
    
    Attributes:
        data_dir: Base data directory.
        train_file: Training data filename.
        validation_file: Validation data filename.
        test_file: Test data filename.
        models_dir: Directory containing boundary models.
        input_boundary_model_file: Input boundary model filename.
        output_boundary_model_file: Output boundary model filename.
        sample_fractions: List of sampling fractions to test.
        n_sampling_repeats: Number of repetitions per fraction.
        training_seed: Fixed seed for model initialization.
        internal_val_ratio: Ratio of internal validation split.
        internal_split_seed: Seed for internal split.
        enable_cleaning: Whether to apply KMeans-LOF cleaning.
        cleaning_config: Configuration for KMeans-LOF cleaner.
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
        device: Computing device ('auto', 'cuda', 'cpu').
        log_level: Logging level.
    """
    data_dir: Path = PROJECT_ROOT / 'data' / 'solubility' / 'fixed_splits'
    train_file: str = 'train.xlsx'
    validation_file: str = 'val.xlsx'
    test_file: str = 'test.xlsx'
    
    models_dir: Path = PROJECT_ROOT / 'models' / 'solubility' / 'binary'
    input_boundary_model_file: str = 'KCl_H2O.pth'
    output_boundary_model_file: str = 'MgCl2_H2O.pth'
    
    sample_fractions: List[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.00]
    )
    n_sampling_repeats: int = 10
    
    training_seed: int = 42
    internal_val_ratio: float = 0.2
    internal_split_seed: int = 999
    
    enable_cleaning: bool = True
    cleaning_config: Dict = field(default_factory=lambda: {
        'lof_contamination': 0.2,
        'lof_n_neighbors': 5,
        'min_cluster_size': 10,
        'auto_select_k': True,
        'min_n_clusters': 2,
        'max_n_clusters': 24,
        'fixed_n_clusters': None,
        'random_state': 42,
        'enable_residual_detection': True,
        'moving_median_window': 5,
        'consensus_weight': 0.5
    })
    
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'small_sample' / 'kmeans_lof'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'kmeans_lof_'
    
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
    config: KMeansLOFSensitivityConfig,
    save_dir: Path,
    sampling_info: Optional[Dict[str, Any]] = None,
    cleaning_info: Optional[Dict[str, Any]] = None,
    internal_split_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        sampling_info: Sampling information dictionary.
        cleaning_info: Cleaning information dictionary.
        internal_split_info: Internal split information dictionary.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Small Sample Sensitivity with KMeans-LOF',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device
        },
        'data_info': {
            'sample_fraction': sampling_info.get('fraction', 'N/A') if sampling_info else 'N/A',
            'sampling_index': sampling_info.get('sampling_idx', 'N/A') if sampling_info else 'N/A',
            'sampling_seed': sampling_info.get('sampling_seed', 'N/A') if sampling_info else 'N/A',
            'n_sampled': sampling_info.get('n_sample', 'N/A') if sampling_info else 'N/A',
            'n_cleaned': cleaning_info.get('n_after_cleaning', 'N/A') if cleaning_info else 'N/A',
            'n_outliers_removed': cleaning_info.get('n_outliers', 'N/A') if cleaning_info else 'N/A'
        },
        'internal_split': {
            'method': 'Train-Val Split',
            'val_ratio': config.internal_val_ratio,
            'split_seed': config.internal_split_seed,
            'n_internal_train': internal_split_info.get('n_internal_train', 'N/A') if internal_split_info else 'N/A',
            'n_internal_val': internal_split_info.get('n_internal_val', 'N/A') if internal_split_info else 'N/A'
        },
        'training_config': {
            'dnn_epochs': config.dnn_epochs,
            'dnn_learning_rate': config.dnn_learning_rate,
            'dnn_layer_dim': config.dnn_layer_dim,
            'dnn_node_dim': config.dnn_node_dim,
            'training_seed': config.training_seed
        },
        'cleaning_config': {
            'enabled': config.enable_cleaning,
            'parameters': config.cleaning_config if config.enable_cleaning else None
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

class SingleKMeansLOFExperimentRunner:
    """
    Runner for a single KMeans-LOF experiment.
    
    Handles complete workflow for one sample configuration:
    - Sample from training pool
    - Apply KMeans-LOF cleaning
    - Split into internal train/val
    - Train DNN with early stopping
    - Evaluate on global val/test sets
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
        config: KMeansLOFSensitivityConfig,
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
    
    def clean_with_kmeans_lof(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply KMeans-LOF cleaning to training data.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
        
        Returns:
            Tuple of (X_clean, y_clean, cleaning_info).
        """
        if not self.config.enable_cleaning:
            self.logger.info("KMeans-LOF cleaning disabled")
            return X_train.copy(), y_train.copy(), {
                'n_outliers': 0,
                'outlier_rate': 0.0,
                'n_before_cleaning': len(X_train),
                'n_after_cleaning': len(X_train)
            }
        
        self.logger.info("="*70)
        self.logger.info("Applying KMeans-LOF Data Cleaning")
        self.logger.info("="*70)
        self.logger.info(f"Data before cleaning: {len(X_train)} samples")
        
        X_with_temp = np.column_stack([X_train, y_train])
        
        cleaner = TKMeansLOF(
            config=self.config.cleaning_config,
            system_type='solubility',
            verbose=False
        )
        
        X_clean_full, y_clean, outlier_indices, clean_info = cleaner.clean(
            X_with_temp, y_train
        )
        
        X_clean = X_clean_full[:, :2]
        
        self.logger.info(f"Data after cleaning: {len(X_clean)} samples")
        self.logger.info(f"Outliers removed: {clean_info['n_outliers']} ({clean_info['outlier_rate']*100:.2f}%)")
        self.logger.info("="*70)
        
        cleaning_info = {
            'n_outliers': clean_info['n_outliers'],
            'outlier_rate': clean_info['outlier_rate'],
            'n_before_cleaning': len(X_train),
            'n_after_cleaning': len(X_clean),
            'n_clusters': clean_info.get('n_clusters', 'N/A'),
            'method': clean_info.get('method', 'KMeans-LOF')
        }
        
        return X_clean, y_clean, cleaning_info
    
    def split_internal_train_val(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Split training data into internal train/val sets.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
        
        Returns:
            Tuple of (X_internal_train, X_internal_val, y_internal_train, y_internal_val, split_info).
        """
        X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
            X_train, y_train,
            test_size=self.config.internal_val_ratio,
            random_state=self.config.internal_split_seed
        )
        
        split_info = {
            'n_internal_train': len(X_internal_train),
            'n_internal_val': len(X_internal_val),
            'split_ratio': self.config.internal_val_ratio,
            'split_seed': self.config.internal_split_seed
        }
        
        self.logger.info("Internal train/val split:")
        self.logger.info(f"  Internal train: {len(X_internal_train)} samples")
        self.logger.info(f"  Internal val: {len(X_internal_val)} samples")
        
        return X_internal_train, X_internal_val, y_internal_train, y_internal_val, split_info
    
    def train_dnn(
        self,
        X_internal_train: np.ndarray,
        y_internal_train: np.ndarray,
        X_internal_val: np.ndarray,
        y_internal_val: np.ndarray
    ) -> Tuple[Dict[str, Any], nn.Module, StandardScaler, StandardScaler]:
        """
        Train DNN on internal train/val sets.
        
        Args:
            X_internal_train: Internal training features.
            y_internal_train: Internal training targets.
            X_internal_val: Internal validation features.
            y_internal_val: Internal validation targets.
        
        Returns:
            Tuple of (result, dnn, xScaler, yScaler).
        """
        device = torch.device(self.config.device)
        
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        
        X_internal_train_scaled = xScaler.fit_transform(X_internal_train)
        y_internal_train_scaled = yScaler.fit_transform(y_internal_train.reshape(-1, 1)).flatten()
        
        X_internal_val_scaled = xScaler.transform(X_internal_val)
        y_internal_val_scaled = yScaler.transform(y_internal_val.reshape(-1, 1)).flatten()
        
        torch.manual_seed(self.config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training_seed)
        
        X_internal_train_tensor = torch.FloatTensor(X_internal_train_scaled).to(device)
        y_internal_train_tensor = torch.FloatTensor(y_internal_train_scaled).to(device)
        X_internal_val_tensor = torch.FloatTensor(X_internal_val_scaled).to(device)
        y_internal_val_tensor = torch.FloatTensor(y_internal_val_scaled).to(device)
        
        input_dim = X_internal_train.shape[1]
        dnn = DNN(
            in_dim=input_dim,
            out_dim=1,
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim
        ).to(device)
        
        optimizer = torch.optim.Adam(dnn.parameters(), lr=self.config.dnn_learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.dnn_epochs):
            dnn.train()
            optimizer.zero_grad()
            y_pred = dnn(X_internal_train_tensor)
            train_loss = criterion(y_pred, y_internal_train_tensor)
            train_loss.backward()
            optimizer.step()
        
        dnn.eval()
        with torch.no_grad():
            y_internal_train_pred_scaled = dnn(X_internal_train_tensor).cpu().numpy()
            y_internal_val_pred_scaled = dnn(X_internal_val_tensor).cpu().numpy()
        
        y_internal_train_pred = yScaler.inverse_transform(y_internal_train_pred_scaled).flatten()
        y_internal_val_pred = yScaler.inverse_transform(y_internal_val_pred_scaled).flatten()
        
        internal_train_metrics = {
            'internal_train_r2': metrics.r2_score(y_internal_train, y_internal_train_pred),
            'internal_train_rmse': np.sqrt(metrics.mean_squared_error(y_internal_train, y_internal_train_pred)),
            'internal_train_mae': metrics.mean_absolute_error(y_internal_train, y_internal_train_pred),
            'internal_val_r2': metrics.r2_score(y_internal_val, y_internal_val_pred),
            'internal_val_rmse': np.sqrt(metrics.mean_squared_error(y_internal_val, y_internal_val_pred)),
            'internal_val_mae': metrics.mean_absolute_error(y_internal_val, y_internal_val_pred)
        }
        
        result = {
            'metrics': internal_train_metrics,
            'predictions': {
                'internal_train': y_internal_train_pred,
                'internal_val': y_internal_val_pred
            },
            'true_values': {
                'internal_train': y_internal_train,
                'internal_val': y_internal_val
            }
        }
        
        return result, dnn, xScaler, yScaler
    
    def evaluate_on_global_sets(
        self,
        dnn: nn.Module,
        xScaler: StandardScaler,
        yScaler: StandardScaler,
        X_train_full: np.ndarray,
        y_train_full: np.ndarray,
        X_val_global: np.ndarray,
        y_val_global: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on global validation and test sets.
        
        Args:
            dnn: Trained model.
            xScaler: Feature scaler.
            yScaler: Target scaler.
            X_train_full: Full sampled and cleaned training features.
            y_train_full: Full sampled and cleaned training targets.
            X_val_global: Global validation features.
            y_val_global: Global validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary with complete metrics, predictions, and true values.
        """
        device = torch.device(self.config.device)
        dnn.eval()
        
        X_train_full_scaled = xScaler.transform(X_train_full)
        with torch.no_grad():
            y_train_full_pred_scaled = dnn(torch.FloatTensor(X_train_full_scaled).to(device)).cpu().numpy()
        y_train_full_pred = yScaler.inverse_transform(y_train_full_pred_scaled).flatten()
        
        X_val_global_scaled = xScaler.transform(X_val_global)
        with torch.no_grad():
            y_val_global_pred_scaled = dnn(torch.FloatTensor(X_val_global_scaled).to(device)).cpu().numpy()
        y_val_global_pred = yScaler.inverse_transform(y_val_global_pred_scaled).flatten()
        
        X_test_scaled = xScaler.transform(X_test)
        with torch.no_grad():
            y_test_pred_scaled = dnn(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
        y_test_pred = yScaler.inverse_transform(y_test_pred_scaled).flatten()
        
        global_metrics = {
            'train_r2': metrics.r2_score(y_train_full, y_train_full_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train_full, y_train_full_pred)),
            'train_mae': metrics.mean_absolute_error(y_train_full, y_train_full_pred),
            'val_r2': metrics.r2_score(y_val_global, y_val_global_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_val_global, y_val_global_pred)),
            'val_mae': metrics.mean_absolute_error(y_val_global, y_val_global_pred),
            'test_r2': metrics.r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'test_mae': metrics.mean_absolute_error(y_test, y_test_pred)
        }
        
        result = {
            'metrics': global_metrics,
            'predictions': {
                'train': y_train_full_pred,
                'val': y_val_global_pred,
                'test': y_test_pred
            },
            'true_values': {
                'train': y_train_full,
                'val': y_val_global,
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
        
        df_train = pd.DataFrame({
            'Temperature': X_train[:, 0],
            'Input_Composition': X_train[:, 1],
            'Output_True': result['true_values']['train'],
            'Output_Pred': result['predictions']['train'],
            'Error': result['true_values']['train'] - result['predictions']['train']
        })
        df_train.to_excel(excel_dir / f'{self.config.excel_prefix}train_predictions.xlsx', index=False)
        
        df_val = pd.DataFrame({
            'Temperature': X_val[:, 0],
            'Input_Composition': X_val[:, 1],
            'Output_True': result['true_values']['val'],
            'Output_Pred': result['predictions']['val'],
            'Error': result['true_values']['val'] - result['predictions']['val']
        })
        df_val.to_excel(excel_dir / f'{self.config.excel_prefix}val_predictions.xlsx', index=False)
        
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
        
        physics_results = self.results.get('physical_evaluation', None)
        
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
        
        evaluator = PhysicalConsistencyEvaluator(
            input_boundary_model=self.input_boundary_model,
            output_boundary_model=self.output_boundary_model,
            boundary_decay_lambda=self.config.boundary_decay_lambda,
            smoothness_decay_lambda=self.config.smoothness_decay_lambda
        )
        
        self.logger.info(f"Running physics evaluation...")
        physics_score, physics_results = evaluator.evaluate_full(
            dnn, xScaler, yScaler, predicted_data, self.config.device
        )
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Physics Evaluation Complete")
        self.logger.info("="*70)
        self.logger.info(f"Overall Physics Score: {physics_score:.6f}")
        
        boundary_score = physics_results.get('boundary_score', 'N/A')
        smoothness_score = physics_results.get('smoothness_score', 'N/A')
        if boundary_score != 'N/A':
            self.logger.info(f"  - Boundary Consistency: {boundary_score:.6f}")
        if smoothness_score != 'N/A':
            self.logger.info(f"  - Thermodynamic Smoothness: {smoothness_score:.6f}")
        
        self.logger.info("="*70)
        
        report = evaluator.generate_evaluation_report(physics_results)
        report_path = self.output_dir / 'data' / 'physical_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Physics evaluation report saved: {report_path}")
        
        self.results['physical_evaluation'] = physics_results
        self.results['physics_score'] = physics_score
        self.results['predicted_data'] = predicted_data
    
    def run(
        self,
        train_pool: Tuple[np.ndarray, np.ndarray],
        val_global: Tuple[np.ndarray, np.ndarray],
        test_set: Tuple[np.ndarray, np.ndarray],
        fraction: float,
        sampling_idx: int
    ) -> Dict[str, Any]:
        """
        Run complete experiment workflow.
        
        Args:
            train_pool: Training pool data (X, y).
            val_global: Global validation data (X, y).
            test_set: Test set data (X, y).
            fraction: Sampling fraction.
            sampling_idx: Sampling repetition index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"Experiment: Fraction={int(fraction*100)}%, Sampling={sampling_idx}")
        self.logger.info("="*70)
        
        X_val_global, y_val_global = val_global
        X_test, y_test = test_set
        
        X_sampled, y_sampled, sampling_seed = sample_from_pool(
            train_pool, fraction, sampling_idx
        )
        
        sampling_info = {
            'fraction': fraction,
            'sampling_idx': sampling_idx,
            'sampling_seed': sampling_seed,
            'n_sample': len(X_sampled)
        }
        
        self.logger.info(f"Data sampling completed:")
        self.logger.info(f"  Sampled: {len(X_sampled)} points ({fraction*100:.0f}%)")
        self.logger.info(f"  Sampling seed: {sampling_seed}")
        
        X_cleaned, y_cleaned, cleaning_info = self.clean_with_kmeans_lof(
            X_sampled, y_sampled
        )
        
        X_internal_train, X_internal_val, y_internal_train, y_internal_val, split_info = self.split_internal_train_val(
            X_cleaned, y_cleaned
        )
        
        self.logger.info("Training DNN with internal validation...")
        result_train, dnn, xScaler, yScaler = self.train_dnn(
            X_internal_train, y_internal_train,
            X_internal_val, y_internal_val
        )
        
        self.logger.info(f"Training completed:")
        self.logger.info(f"  Internal train R²: {result_train['metrics']['internal_train_r2']:.6f}")
        self.logger.info(f"  Internal val R²: {result_train['metrics']['internal_val_r2']:.6f}")
        
        self.logger.info("Evaluating on global validation and test sets...")
        result_global = self.evaluate_on_global_sets(
            dnn, xScaler, yScaler,
            X_cleaned, y_cleaned,
            X_val_global, y_val_global,
            X_test, y_test
        )
        
        self.logger.info(f"Global evaluation completed:")
        self.logger.info(f"  Train R² (full sampled): {result_global['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (global): {result_global['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R²: {result_global['metrics']['test_r2']:.6f}")
        
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
        
        if self.config.save_predictions:
            self.logger.info("Saving predictions...")
            self.save_predictions_to_excel(result_global, X_cleaned, X_val_global, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result_global)
        
        save_config(
            result_global['metrics'],
            self.config,
            self.output_dir,
            sampling_info,
            cleaning_info,
            split_info
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        return_dict = {
            'fraction': fraction,
            'sampling_idx': sampling_idx,
            'sampling_seed': sampling_seed,
            'n_sampled': len(X_sampled),
            'n_cleaned': len(X_cleaned),
            'n_outliers': cleaning_info['n_outliers'],
            'n_internal_train': split_info['n_internal_train'],
            'n_internal_val': split_info['n_internal_val'],
            'metrics': result_global['metrics'],
            'physics_score': self.results.get('physics_score'),
            'physics_boundary': None,
            'physics_smoothness': None,
            'elapsed_time': elapsed_time
        }
        
        if self.results.get('physical_evaluation'):
            phys_results = self.results['physical_evaluation']
            return_dict['physics_boundary'] = phys_results.get('boundary_score')
            return_dict['physics_smoothness'] = phys_results.get('smoothness_score')
        
        return return_dict


# ==============================================================================
# Experiment Manager
# ==============================================================================

class KMeansLOFSensitivityExperimentManager:
    """
    Manager for complete KMeans-LOF sample sensitivity study.
    
    Coordinates multiple experiments across different sample sizes:
    - Loads fixed train/val/test sets
    - Loads boundary models for physics evaluation
    - Runs experiments for each sample fraction with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_pool: Training pool data.
        val_global: Global validation data.
        test_set: Test set data.
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: KMeansLOFSensitivityConfig) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info("Loading datasets...")
        self.train_pool = self._load_train_pool()
        self.val_global = self._load_val_global()
        self.test_set = self._load_test_set()
        
        self.logger.info(f"Data loaded:")
        self.logger.info(f"  Training pool: {len(self.train_pool[0])} samples")
        self.logger.info(f"  Global validation: {len(self.val_global[0])} samples")
        self.logger.info(f"  Test: {len(self.test_set[0])} samples")
        
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        self.all_results: Dict[float, List[Dict[str, Any]]] = {
            fraction: [] for fraction in config.sample_fractions
        }
    
    def _load_train_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training pool data."""
        train_path = self.config.data_dir / self.config.train_file
        return load_ternary_data(train_path)
    
    def _load_val_global(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load global validation data."""
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
        """Run all sample sensitivity experiments."""
        total_experiments = len(self.config.sample_fractions) * self.config.n_sampling_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Starting KMeans-LOF Sample Sensitivity Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}")
        self.logger.info(f"Repeats per fraction: {self.config.n_sampling_repeats}")
        self.logger.info(f"KMeans-LOF cleaning: {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        
        experiment_count = 0
        
        for fraction in self.config.sample_fractions:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Running experiments for {int(fraction*100)}% sampling")
            self.logger.info("="*70)
            
            for sampling_idx in range(self.config.n_sampling_repeats):
                experiment_count += 1
                
                output_dir = (
                    self.config.output_dir /
                    f'fraction_{int(fraction*100):03d}' /
                    f'repeat_{sampling_idx:02d}'
                )
                
                runner = SingleKMeansLOFExperimentRunner(
                    self.config,
                    output_dir,
                    self.input_boundary_model,
                    self.output_boundary_model
                )
                
                result = runner.run(
                    self.train_pool,
                    self.val_global,
                    self.test_set,
                    fraction,
                    sampling_idx
                )
                
                self.all_results[fraction].append(result)
                
                self.logger.info(f"Progress: {experiment_count}/{total_experiments} completed")
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Saving aggregated results")
        self.logger.info("="*70)
        
        self._save_aggregated_results()
        self._generate_summary_report()
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("KMeans-LOF Sample Sensitivity Study completed successfully")
        self.logger.info("="*70)
    
    def _save_aggregated_results(self) -> None:
        """Save aggregated results across all experiments."""
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        data = []
        
        for fraction in self.config.sample_fractions:
            results = self.all_results[fraction]
            
            if len(results) == 0:
                continue
            
            row = {
                'Sample_Fraction': f'{int(fraction*100)}%',
                'n_repeats': len(results)
            }
            
            n_sampled_values = [r['n_sampled'] for r in results]
            row['n_sampled_mean'] = np.mean(n_sampled_values)
            
            n_cleaned_values = [r['n_cleaned'] for r in results]
            row['n_cleaned_mean'] = np.mean(n_cleaned_values)
            
            n_outliers_values = [r['n_outliers'] for r in results]
            row['n_outliers_mean'] = np.mean(n_outliers_values)
            
            test_r2_values = [r['metrics']['test_r2'] for r in results]
            row['Test_R2_Mean'] = np.mean(test_r2_values)
            row['Test_R2_Std'] = np.std(test_r2_values, ddof=1)
            
            rmse_values = [r['metrics']['test_rmse'] for r in results]
            row['Test_RMSE_Mean'] = np.mean(rmse_values)
            row['Test_RMSE_Std'] = np.std(rmse_values, ddof=1)
            
            mae_values = [r['metrics']['test_mae'] for r in results]
            row['Test_MAE_Mean'] = np.mean(mae_values)
            row['Test_MAE_Std'] = np.std(mae_values, ddof=1)
            
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
        report.append("KMeans-LOF Sample Sensitivity Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Sample fractions: {[f'{int(f*100)}%' for f in self.config.sample_fractions]}")
        report.append(f"Repeats per fraction: {self.config.n_sampling_repeats}")
        report.append(f"Total experiments: {len(self.config.sample_fractions) * self.config.n_sampling_repeats}")
        report.append(f"KMeans-LOF cleaning: {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Method: Random sampling with KMeans-LOF cleaning")
        report.append(f"Internal validation ratio: {self.config.internal_val_ratio}")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        report.append(f"Evaluation: Clean global validation and test sets")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training pool: {len(self.train_pool[0])} samples")
        report.append(f"Global validation: {len(self.val_global[0])} samples")
        report.append(f"Test: {len(self.test_set[0])} samples")
        
        report.append("\n" + "-"*70)
        report.append("Results Summary (Test R²)")
        report.append("-"*70)
        
        for fraction in self.config.sample_fractions:
            results = self.all_results[fraction]
            if len(results) == 0:
                continue
            
            r2_values = [r['metrics']['test_r2'] for r in results]
            
            mean_r2 = np.mean(r2_values)
            std_r2 = np.std(r2_values, ddof=1)
            
            report.append(
                f"{int(fraction*100):3d}% sample: "
                f"R² = {mean_r2:.6f} ± {std_r2:.6f}"
            )
        
        report.append("\n" + "="*70)
        
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
    Main entry point for KMeans-LOF sample sensitivity experiment.
    
    Configures and runs complete sample sensitivity study with KMeans-LOF cleaning.
    """
    config = KMeansLOFSensitivityConfig()
    
    config.sample_fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    config.n_sampling_repeats = 20
    config.training_seed = 42
    config.internal_val_ratio = 0.2
    config.internal_split_seed = 999
    
    config.enable_cleaning = True
    config.cleaning_config = {
        'lof_contamination': 0.2,
        'lof_n_neighbors': 5,
        'min_cluster_size': 10,
        'auto_select_k': True,
        'min_n_clusters': 2,
        'max_n_clusters': 24,
        'fixed_n_clusters': None,
        'random_state': 42,
        'enable_residual_detection': True,
        'moving_median_window': 5,
        'consensus_weight': 0.5
    }
    
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
    logger.info(f"  - Train: {config.data_dir / config.train_file}")
    logger.info(f"  - Val: {config.data_dir / config.validation_file}")
    logger.info(f"  - Test: {config.data_dir / config.test_file}")
    logger.info(f"Models directory: {config.models_dir}")
    logger.info(f"  - Input boundary: {config.models_dir / config.input_boundary_model_file}")
    logger.info(f"  - Output boundary: {config.models_dir / config.output_boundary_model_file}")
    logger.info(f"Output directory: {config.output_dir}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("KMeans-LOF Sample Sensitivity Experiment - Baseline")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Sample fractions: {[f'{int(f*100)}%' for f in config.sample_fractions]}")
    logger.info(f"Repeats per fraction: {config.n_sampling_repeats}")
    logger.info(f"Total experiments: {len(config.sample_fractions) * config.n_sampling_repeats}")
    logger.info("")
    logger.info("Data files:")
    logger.info(f"  Training pool: {config.train_file}")
    logger.info(f"  Global validation: {config.validation_file}")
    logger.info(f"  Test set: {config.test_file}")
    logger.info("")
    logger.info("Key features:")
    logger.info(f"  - KMeans-LOF cleaning: {'Enabled' if config.enable_cleaning else 'Disabled'}")
    logger.info(f"  - Internal split: {config.internal_val_ratio*100:.0f}% validation")
    logger.info(f"  - Training seed: {config.training_seed} (fixed for reproducibility)")
    logger.info(f"  - Physics evaluation: boundary_lambda={config.boundary_decay_lambda}, smoothness_lambda={config.smoothness_decay_lambda}")
    
    manager = KMeansLOFSensitivityExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("KMeans-LOF sample sensitivity experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()