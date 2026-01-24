"""
Noise robustness experiment with T-KMeans-LOF outlier detection.

Evaluates DNN robustness to label noise when combined with KMeans-LOF data
cleaning. This experiment assesses whether data cleaning can mitigate the
negative impact of noisy training labels on model performance.

Experimental Design:
    1. Load fixed train/val/test sets
    2. For each noise level (0%, 5%, 10%, 15%, 20%):
        - Repeat N times:
            - Add Gaussian noise to training labels
            - Apply KMeans-LOF cleaning to noisy data
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
from sklearn import metrics
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
    
    Noise is proportional to the range of target values:
        noise_std = noise_level * (y_max - y_min)
    
    Args:
        train_set: Tuple of (X, y) training data.
        noise_level: Noise level as fraction (0.0-1.0).
        noise_idx: Noise repetition index.
        base_seed: Base random seed.
    
    Returns:
        Tuple of (X_train, y_noisy, noise_seed).
    """
    X_train, y_train = train_set
    
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed
    
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    noise_std = noise_level * y_range
    
    noise = np.random.normal(0, noise_std, size=len(y_train))
    y_noisy = y_train + noise
    
    return X_train.copy(), y_noisy, noise_seed


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class KMeansLOFNoiseRobustnessConfig:
    """
    Configuration for KMeans-LOF noise robustness experiment.
    
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
        training_seed: Fixed seed for model initialization.
        noise_base_seed: Base seed for noise injection.
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
    
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20]
    )
    n_noise_repeats: int = 10
    
    training_seed: int = 42
    noise_base_seed: int = 20000
    
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
    
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'noise' / 'noise_robustness_T_KMeans_LOF_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'kmeans_lof_noisy_'
    
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: auto-detect device."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    config: KMeansLOFNoiseRobustnessConfig,
    save_dir: Path,
    noise_info: Optional[Dict[str, Any]] = None,
    cleaning_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        noise_info: Noise information dictionary.
        cleaning_info: Cleaning information dictionary.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Noise Robustness with KMeans-LOF',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device
        },
        'noise_info': {
            'noise_level': noise_info.get('noise_level', 'N/A') if noise_info else 'N/A',
            'noise_index': noise_info.get('noise_idx', 'N/A') if noise_info else 'N/A',
            'noise_seed': noise_info.get('noise_seed', 'N/A') if noise_info else 'N/A',
            'noise_std': noise_info.get('noise_std', 'N/A') if noise_info else 'N/A'
        },
        'cleaning_info': {
            'enabled': config.enable_cleaning,
            'n_before_cleaning': cleaning_info.get('n_before_cleaning', 'N/A') if cleaning_info else 'N/A',
            'n_after_cleaning': cleaning_info.get('n_after_cleaning', 'N/A') if cleaning_info else 'N/A',
            'n_outliers': cleaning_info.get('n_outliers', 'N/A') if cleaning_info else 'N/A',
            'outlier_rate': cleaning_info.get('outlier_rate', 'N/A') if cleaning_info else 'N/A'
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
    
    config_dict = convert_numpy_types(config_dict)
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


# ==============================================================================
# Single Experiment Runner
# ==============================================================================

class SingleKMeansLOFNoiseExperimentRunner:
    """
    Runner for a single KMeans-LOF noise experiment.
    
    Handles complete workflow for one noise configuration:
    - Add noise to training labels
    - Apply KMeans-LOF cleaning
    - Train DNN with fixed seed
    - Evaluate on clean val/test sets
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
        config: KMeansLOFNoiseRobustnessConfig,
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
        Apply KMeans-LOF cleaning to noisy training data.
        
        Args:
            X_train: Training features.
            y_train: Noisy training targets.
        
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
    
    def train_dnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, Any], nn.Module, StandardScaler, StandardScaler]:
        """
        Train DNN on cleaned noisy data and evaluate on clean sets.
        
        Args:
            X_train: Cleaned training features.
            y_train: Cleaned training targets (originally noisy).
            X_val: Clean validation features.
            y_val: Clean validation targets.
            X_test: Clean test features.
            y_test: Clean test targets.
        
        Returns:
            Tuple of (result, dnn, xScaler, yScaler).
        """
        device = torch.device(self.config.device)
        
        xScaler = StandardScaler()
        yScaler = StandardScaler()
        
        X_train_scaled = xScaler.fit_transform(X_train)
        y_train_scaled = yScaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = xScaler.transform(X_val)
        y_val_scaled = yScaler.transform(y_val.reshape(-1, 1)).flatten()
        
        X_test_scaled = xScaler.transform(X_test)
        y_test_scaled = yScaler.transform(y_test.reshape(-1, 1)).flatten()
        
        torch.manual_seed(self.config.training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.training_seed)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val_scaled).to(device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
        
        input_dim = X_train.shape[1]
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
            y_pred = dnn(X_train_tensor)
            train_loss = criterion(y_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()
        
        dnn.eval()
        with torch.no_grad():
            y_train_pred_scaled = dnn(X_train_tensor).cpu().numpy()
            y_val_pred_scaled = dnn(X_val_tensor).cpu().numpy()
            y_test_pred_scaled = dnn(X_test_tensor).cpu().numpy()
        
        y_train_pred = yScaler.inverse_transform(y_train_pred_scaled).flatten()
        y_val_pred = yScaler.inverse_transform(y_val_pred_scaled).flatten()
        y_test_pred = yScaler.inverse_transform(y_test_pred_scaled).flatten()
        
        train_metrics = {
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
            'metrics': train_metrics,
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
        
        return result, dnn, xScaler, yScaler
    
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
        train_set: Tuple[np.ndarray, np.ndarray],
        val_set: Tuple[np.ndarray, np.ndarray],
        test_set: Tuple[np.ndarray, np.ndarray],
        noise_level: float,
        noise_idx: int
    ) -> Dict[str, Any]:
        """
        Run complete experiment workflow.
        
        Args:
            train_set: Training data (X, y).
            val_set: Validation data (X, y).
            test_set: Test set data (X, y).
            noise_level: Noise level (0.0-1.0).
            noise_idx: Noise repetition index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"Experiment: Noise={int(noise_level*100)}%, Repeat={noise_idx}")
        self.logger.info("="*70)
        
        X_val, y_val = val_set
        X_test, y_test = test_set
        
        X_train_noisy, y_train_noisy, noise_seed = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed
        )
        
        y_range = train_set[1].max() - train_set[1].min()
        noise_std = noise_level * y_range
        
        noise_info = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'noise_std': noise_std
        }
        
        self.logger.info(f"Noise injection completed:")
        self.logger.info(f"  Noise level: {noise_level*100:.0f}%")
        self.logger.info(f"  Noise std: {noise_std:.4f}")
        self.logger.info(f"  Noise seed: {noise_seed}")
        
        X_cleaned, y_cleaned, cleaning_info = self.clean_with_kmeans_lof(
            X_train_noisy, y_train_noisy
        )
        
        self.logger.info("Training DNN on cleaned noisy data...")
        result, dnn, xScaler, yScaler = self.train_dnn(
            X_cleaned, y_cleaned, X_val, y_val, X_test, y_test
        )
        
        self.logger.info(f"Training completed:")
        self.logger.info(f"  Train R² (on cleaned noisy): {result['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (clean): {result['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R² (clean): {result['metrics']['test_r2']:.6f}")
        
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
            self.save_predictions_to_excel(result, X_cleaned, X_val, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result)
        
        save_config(
            result['metrics'],
            self.config,
            self.output_dir,
            noise_info,
            cleaning_info
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        return_dict = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'noise_std': noise_std,
            'n_before_cleaning': len(X_train_noisy),
            'n_after_cleaning': len(X_cleaned),
            'n_outliers': cleaning_info['n_outliers'],
            'metrics': result['metrics'],
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

class KMeansLOFNoiseRobustnessExperimentManager:
    """
    Manager for complete KMeans-LOF noise robustness study.
    
    Coordinates multiple experiments across different noise levels:
    - Loads fixed train/val/test sets
    - Loads boundary models for physics evaluation
    - Runs experiments for each noise level with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_set: Training set data.
        val_set: Validation data.
        test_set: Test set data.
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: KMeansLOFNoiseRobustnessConfig) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info("Loading datasets...")
        self.train_set = self._load_train_set()
        self.val_set = self._load_val_set()
        self.test_set = self._load_test_set()
        
        self.logger.info(f"Data loaded:")
        self.logger.info(f"  Training: {len(self.train_set[0])} samples")
        self.logger.info(f"  Validation: {len(self.val_set[0])} samples")
        self.logger.info(f"  Test: {len(self.test_set[0])} samples")
        
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        self.all_results: Dict[float, List[Dict[str, Any]]] = {
            noise_level: [] for noise_level in config.noise_levels
        }
    
    def _load_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training set data."""
        train_path = self.config.data_dir / self.config.train_file
        return load_ternary_data(train_path)
    
    def _load_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load validation data."""
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
        """Run all noise robustness experiments."""
        total_experiments = len(self.config.noise_levels) * self.config.n_noise_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Starting KMeans-LOF Noise Robustness Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        self.logger.info(f"Repeats per level: {self.config.n_noise_repeats}")
        self.logger.info(f"KMeans-LOF cleaning: {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        
        experiment_count = 0
        
        for noise_level in self.config.noise_levels:
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Running experiments for {int(noise_level*100)}% noise")
            self.logger.info("="*70)
            
            for noise_idx in range(self.config.n_noise_repeats):
                experiment_count += 1
                
                output_dir = (
                    self.config.output_dir /
                    f'noise_{int(noise_level*100):03d}' /
                    f'repeat_{noise_idx:02d}'
                )
                
                runner = SingleKMeansLOFNoiseExperimentRunner(
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
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Saving aggregated results")
        self.logger.info("="*70)
        
        self._save_aggregated_results()
        self._generate_summary_report()
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("KMeans-LOF Noise Robustness Study completed successfully")
        self.logger.info("="*70)
    
    def _save_aggregated_results(self) -> None:
        """Save aggregated results across all experiments."""
        summary_dir = self.config.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True, parents=True)
        
        data = []
        
        for noise_level in self.config.noise_levels:
            results = self.all_results[noise_level]
            
            if len(results) == 0:
                continue
            
            row = {
                'Noise_Level': f'{int(noise_level*100)}%',
                'n_repeats': len(results)
            }
            
            n_before_values = [r['n_before_cleaning'] for r in results]
            row['n_before_cleaning_mean'] = np.mean(n_before_values)
            
            n_after_values = [r['n_after_cleaning'] for r in results]
            row['n_after_cleaning_mean'] = np.mean(n_after_values)
            
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
        report.append("KMeans-LOF Noise Robustness Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        report.append(f"Repeats per level: {self.config.n_noise_repeats}")
        report.append(f"Total experiments: {len(self.config.noise_levels) * self.config.n_noise_repeats}")
        report.append(f"KMeans-LOF cleaning: {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Method: Gaussian noise injection with KMeans-LOF cleaning")
        report.append(f"Training: Noisy data (after cleaning)")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        report.append(f"Evaluation: Clean global validation and test sets")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training: {len(self.train_set[0])} samples")
        report.append(f"Validation: {len(self.val_set[0])} samples")
        report.append(f"Test: {len(self.test_set[0])} samples")
        
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
    Main entry point for KMeans-LOF noise robustness experiment.
    
    Configures and runs complete noise robustness study with KMeans-LOF cleaning.
    """
    config = KMeansLOFNoiseRobustnessConfig()
    
    config.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    config.n_noise_repeats = 10
    config.training_seed = 42
    config.noise_base_seed = 20000
    
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
    logger.info("KMeans-LOF Noise Robustness Experiment")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Noise levels: {[f'{int(n*100)}%' for n in config.noise_levels]}")
    logger.info(f"Repeats per level: {config.n_noise_repeats}")
    logger.info(f"Total experiments: {len(config.noise_levels) * config.n_noise_repeats}")
    logger.info("")
    logger.info("Data files:")
    logger.info(f"  Training: {config.train_file}")
    logger.info(f"  Validation: {config.validation_file}")
    logger.info(f"  Test set: {config.test_file}")
    logger.info("")
    logger.info("Key features:")
    logger.info(f"  - KMeans-LOF cleaning: {'Enabled' if config.enable_cleaning else 'Disabled'}")
    logger.info(f"  - Noise base seed: {config.noise_base_seed}")
    logger.info(f"  - Training seed: {config.training_seed} (fixed for reproducibility)")
    logger.info(f"  - Physics evaluation: boundary_lambda={config.boundary_decay_lambda}, smoothness_lambda={config.smoothness_decay_lambda}")
    
    manager = KMeansLOFNoiseRobustnessExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("KMeans-LOF noise robustness experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()