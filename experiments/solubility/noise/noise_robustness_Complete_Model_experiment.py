"""
Noise robustness experiment with complete M4 pipeline.

Evaluates complete M4 pipeline (TKMeansLOF + IADAF + LDPC) performance 
under label noise. This experiment assesses how the full pipeline handles
noisy training data compared to baseline approaches.

Experimental Design:
    1. Load fixed train/val/test sets
    2. For each noise level (0%, 5%, 10%, 15%, 20%):
        - Repeat N times:
            - Inject Gaussian noise into training labels
            - Split noisy training data into internal train/val
            - Run M4 pipeline (clean → augment → train with LDPC)
            - Evaluate pipeline output model on clean global val/test sets
            - Run physics evaluation
    3. Aggregate results across noise levels

M4 Pipeline Components:
    - TKMeansLOF: Temperature-based KMeans + LOF outlier detection
    - IADAF: Intelligent Adaptive Data Augmentation Framework (Bayesian-optimized WGAN-GP)
    - LDPC: Low-Dimensional Physical Constraints (boundary correction)

Key Features:
    - Noise only affects training labels (not features)
    - Internal validation for M4 pipeline early stopping
    - Global val/test remain clean (independent evaluation)
    - M4 cleaning may help remove some noise effects
    - IADAF augmentation may dilute noise with synthetic data
    - LDPC correction enforces physical boundaries
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
    PhysicalConsistencyEvaluator,
)
from binary_predictor import BinaryPredictor
from solubility_pipeline import SolubilityPipeline, SolubilityPipelineConfig

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
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Add Gaussian noise to training labels.
    
    Args:
        train_set: Tuple of (X_train, y_train).
        noise_level: Noise level as fraction of label range (0.0-1.0).
        noise_idx: Noise repetition index for seed generation.
        base_seed: Base random seed for noise generation.
    
    Returns:
        Tuple of (X_train, y_noisy, noise_seed, noise_std):
            - X_train: Training features (unchanged).
            - y_noisy: Training labels with added noise.
            - noise_seed: Random seed used for noise generation.
            - noise_std: Standard deviation of noise.
    
    Notes:
        - Noise standard deviation: noise_std = noise_level * (y_max - y_min)
        - Noise seed: base_seed + int(noise_level * 10000) + noise_idx * 100
        - For noise_level=0.0, returns original labels unchanged
    """
    X_train, y_train = train_set
    
    noise_seed = base_seed + int(noise_level * 10000) + noise_idx * 100
    
    if noise_level == 0.0:
        return X_train.copy(), y_train.copy(), noise_seed, 0.0
    
    np.random.seed(noise_seed)
    y_range = y_train.max() - y_train.min()
    noise_std = noise_level * y_range
    
    noise = np.random.normal(0, noise_std, size=len(y_train))
    y_noisy = y_train + noise
    
    return X_train.copy(), y_noisy, noise_seed, noise_std


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class M4NoiseRobustnessConfig:
    """
    Configuration for M4 complete pipeline noise robustness experiment.
    
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
        noise_base_seed: Base seed for noise generation.
        internal_val_ratio: Ratio of internal validation split.
        internal_split_seed: Seed for internal split.
        enable_cleaning: Whether to enable TKMeansLOF cleaning.
        n_samples_to_generate: Number of synthetic samples for IADAF.
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
    internal_val_ratio: float = 0.2
    internal_split_seed: int = 999
    
    enable_cleaning: bool = True
    n_samples_to_generate: int = 2000
    
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'noise' / 'm4'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'm4_noisy_'
    
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
    config: M4NoiseRobustnessConfig,
    save_dir: Path,
    noise_info: Optional[Dict[str, Any]] = None,
    internal_split_info: Optional[Dict[str, Any]] = None,
    cleaning_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment configuration to JSON.
    
    Args:
        metrics: Evaluation metrics dictionary.
        config: Experiment configuration.
        save_dir: Directory to save configuration.
        noise_info: Noise injection information dictionary.
        internal_split_info: Internal split information dictionary.
        cleaning_info: Data cleaning information dictionary.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Noise Robustness with M4 Pipeline',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device
        },
        'noise_info': {
            'noise_level': noise_info.get('noise_level', 'N/A') if noise_info else 'N/A',
            'noise_index': noise_info.get('noise_idx', 'N/A') if noise_info else 'N/A',
            'noise_seed': noise_info.get('noise_seed', 'N/A') if noise_info else 'N/A',
            'noise_std': noise_info.get('noise_std', 'N/A') if noise_info else 'N/A'
        },
        'internal_split': {
            'method': 'Train-Val Split',
            'val_ratio': config.internal_val_ratio,
            'split_seed': config.internal_split_seed,
            'n_internal_train': internal_split_info.get('n_internal_train', 'N/A') if internal_split_info else 'N/A',
            'n_internal_val': internal_split_info.get('n_internal_val', 'N/A') if internal_split_info else 'N/A'
        },
        'm4_pipeline': {
            'module_1_cleaning': {
                'enabled': config.enable_cleaning,
                'method': 'TKMeansLOF',
                'n_original': cleaning_info.get('n_original', 'N/A') if cleaning_info else 'N/A',
                'n_cleaned': cleaning_info.get('n_cleaned', 'N/A') if cleaning_info else 'N/A',
                'n_outliers': cleaning_info.get('n_outliers', 'N/A') if cleaning_info else 'N/A'
            },
            'module_2_augmentation': {
                'method': 'IADAF',
                'n_synthetic': config.n_samples_to_generate
            },
            'module_3_ldpc': {
                'method': 'ConformingCorrectedModel',
                'epochs': config.dnn_epochs,
                'learning_rate': config.dnn_learning_rate,
                'layer_dim': config.dnn_layer_dim,
                'node_dim': config.dnn_node_dim
            }
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

class SingleM4NoiseExperimentRunner:
    """
    Runner for a single M4 complete pipeline noise experiment.
    
    Handles complete workflow for one noise configuration:
    - Inject Gaussian noise into training labels
    - Split noisy training data into internal train/val
    - Run M4 pipeline (TKMeansLOF → IADAF → LDPC)
    - Evaluate on clean global val/test sets
    - Physics consistency evaluation
    - Results saving
    
    Attributes:
        config: Experiment configuration.
        output_dir: Output directory for this experiment.
        logger: Logger instance.
        results: Dictionary storing experiment results.
        input_boundary_model: Input boundary model for LDPC and physics.
        output_boundary_model: Output boundary model for LDPC and physics.
    """
    
    def __init__(
        self,
        config: M4NoiseRobustnessConfig,
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
    
    def split_internal_train_val(
        self,
        X_train_noisy: np.ndarray,
        y_train_noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Split noisy training data into internal train/val sets for M4 pipeline.
        
        Args:
            X_train_noisy: Training features (no noise in features).
            y_train_noisy: Training labels with injected noise.
        
        Returns:
            Tuple of (X_internal_train, X_internal_val, y_internal_train, y_internal_val, split_info).
        """
        X_internal_train, X_internal_val, y_internal_train, y_internal_val = train_test_split(
            X_train_noisy, y_train_noisy,
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
    
    def create_pipeline_config(self) -> SolubilityPipelineConfig:
        """
        Create M4 pipeline configuration.
        
        Returns:
            SolubilityPipelineConfig instance.
        """
        input_model_path = None
        output_model_path = None
        
        if self.input_boundary_model is not None:
            input_model_path = self.config.models_dir / self.config.input_boundary_model_file
        if self.output_boundary_model is not None:
            output_model_path = self.config.models_dir / self.config.output_boundary_model_file
        
        pipeline_config = SolubilityPipelineConfig(
            use_kfold=False,
            training_seed=self.config.training_seed,
            
            enable_cleaning=self.config.enable_cleaning,
            kmeans_lof_config={
                'lof_contamination': 0.20,
                'lof_n_neighbors': 5,
                'min_cluster_size': 10,
                'auto_select_k': True,
                'min_n_clusters': 5,
                'max_n_clusters': 24,
                'random_state': 42
            },
            
            n_samples_to_generate=self.config.n_samples_to_generate,
            iadaf_config={
                'LATENT_DIM': 10,
                'HIDDEN_DIM': 256,
                'LAMBDA_GP': 10.0,
                'N_CRITIC': 5,
                'DROPOUT': 0.1,
                'LEARNING_RATE_G': 1e-4,
                'LEARNING_RATE_C': 1e-4,
                'BATCH_SIZE': 64,
                'WGAN_EPOCHS': 500,
                'BO_WGAN_EPOCHS': 50,
                'BO_N_ITERATIONS': 100,
                'BO_INIT_POINTS': 10,
                'BO_DNN_EPOCHS': 200,
                'XGB_ERROR_THRESHOLD': 15.0,
                'XGB_MAX_DEPTH': 6,
                'XGB_N_ESTIMATORS': 100,
                'XGB_LEARNING_RATE': 0.1,
                'TRAIN_RATIO': 0.8,
                'VAL_RATIO': 0.2,
                'RANDOM_SEED': self.config.training_seed,
                'VERBOSE': False,
                'DEVICE': self.config.device,
                'LAMBDA_GP_RANGE': (0.05, 0.5),
                'LATENT_DIM_RANGE': (1, 100),
                'HIDDEN_DIM_RANGE': (150, 350),
            },
            
            ldpc_dnn_layers=self.config.dnn_layer_dim,
            ldpc_dnn_nodes=self.config.dnn_node_dim,
            ldpc_epochs=self.config.dnn_epochs,
            ldpc_lr=self.config.dnn_learning_rate,
            
            input_boundary_model_path=input_model_path,
            output_boundary_model_path=output_model_path,
            input_boundary_temp_range=(-40.0, 200.0),
            output_boundary_temp_range=(-40.0, 200.0),
            input_boundary_output_range=(0.0, 80.0),
            output_boundary_output_range=(0.0, 80.0),
            
            T_min=-40.0,
            T_max=200.0,
            device=self.config.device,
            verbose=False
        )
        
        return pipeline_config
    
    def run_m4_pipeline(
        self,
        X_internal_train: np.ndarray,
        y_internal_train: np.ndarray,
        X_internal_val: np.ndarray,
        y_internal_val: np.ndarray
    ) -> Tuple[SolubilityPipeline, Dict[str, Any]]:
        """
        Run M4 complete pipeline.
        
        Args:
            X_internal_train: Internal training features.
            y_internal_train: Internal training targets (with noise).
            X_internal_val: Internal validation features.
            y_internal_val: Internal validation targets (with noise).
        
        Returns:
            Tuple of (trained_pipeline, cleaning_info).
        """
        self.logger.info("Initializing M4 pipeline...")
        
        pipeline_config = self.create_pipeline_config()
        pipeline = SolubilityPipeline(pipeline_config)
        
        self.logger.info("Running M4 pipeline (TKMeansLOF → IADAF → LDPC)...")
        pipeline.fit(X_internal_train, y_internal_train, X_internal_val, y_internal_val)
        
        cleaning_info = {}
        if pipeline.cleaner is not None:
            cleaning_info = {
                'n_original': len(X_internal_train),
                'n_cleaned': len(pipeline.X_cleaned) if pipeline.X_cleaned is not None else 0,
                'n_outliers': len(X_internal_train) - (len(pipeline.X_cleaned) if pipeline.X_cleaned is not None else len(X_internal_train))
            }
            self.logger.info(f"Data cleaning: {cleaning_info['n_original']} → {cleaning_info['n_cleaned']} samples")
        
        self.logger.info("M4 pipeline training completed")
        
        return pipeline, cleaning_info
    
    def evaluate_on_global_sets(
        self,
        pipeline: SolubilityPipeline,
        X_train_noisy: np.ndarray,
        y_train_noisy: np.ndarray,
        X_val_clean: np.ndarray,
        y_val_clean: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate M4 pipeline on clean global sets.
        
        Args:
            pipeline: Trained M4 pipeline.
            X_train_noisy: Full noisy training features.
            y_train_noisy: Full noisy training targets.
            X_val_clean: Clean global validation features.
            y_val_clean: Clean global validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary with complete metrics, predictions, and true values.
        """
        self.logger.info("Evaluating on global validation and test sets...")
        
        y_train_noisy_pred = pipeline.predict(X_train_noisy)
        y_val_clean_pred = pipeline.predict(X_val_clean)
        y_test_pred = pipeline.predict(X_test)
        
        global_metrics = {
            'train_r2': metrics.r2_score(y_train_noisy, y_train_noisy_pred),
            'train_rmse': np.sqrt(metrics.mean_squared_error(y_train_noisy, y_train_noisy_pred)),
            'train_mae': metrics.mean_absolute_error(y_train_noisy, y_train_noisy_pred),
            'val_r2': metrics.r2_score(y_val_clean, y_val_clean_pred),
            'val_rmse': np.sqrt(metrics.mean_squared_error(y_val_clean, y_val_clean_pred)),
            'val_mae': metrics.mean_absolute_error(y_val_clean, y_val_clean_pred),
            'test_r2': metrics.r2_score(y_test, y_test_pred),
            'test_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
            'test_mae': metrics.mean_absolute_error(y_test, y_test_pred)
        }
        
        result = {
            'metrics': global_metrics,
            'predictions': {
                'train': y_train_noisy_pred,
                'val': y_val_clean_pred,
                'test': y_test_pred
            },
            'true_values': {
                'train': y_train_noisy,
                'val': y_val_clean,
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
        pipeline: SolubilityPipeline,
        X_train: np.ndarray
    ) -> None:
        """
        Run physical consistency evaluation on M4 pipeline output.
        
        Args:
            pipeline: Trained M4 pipeline.
            X_train: Training data for temperature range inference.
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
        
        y_phase_pred = pipeline.predict(X_phase)
        
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
        
        model = pipeline.get_trained_model()
        x_scaler, y_scaler = pipeline.get_scalers()
        
        physics_score, physics_results = evaluator.evaluate_full(
            model, x_scaler, y_scaler, predicted_data, self.config.device
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
        Run complete M4 noise experiment workflow.
        
        Args:
            train_set: Training set data (X, y) - clean.
            val_set: Validation set data (X, y) - clean.
            test_set: Test set data (X, y) - clean.
            noise_level: Noise level to inject.
            noise_idx: Noise repetition index.
        
        Returns:
            Dictionary with complete experiment results.
        """
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info(f"M4 Experiment: Noise={int(noise_level*100)}%, Repeat={noise_idx}")
        self.logger.info("="*70)
        
        X_train_clean, y_train_clean = train_set
        X_val_clean, y_val_clean = val_set
        X_test, y_test = test_set
        
        self.logger.info("Injecting Gaussian noise into training labels...")
        X_train_noisy, y_train_noisy, noise_seed, noise_std = add_gaussian_noise(
            train_set, noise_level, noise_idx, self.config.noise_base_seed
        )
        
        noise_info = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'noise_std': noise_std,
            'n_train': len(X_train_noisy)
        }
        
        self.logger.info(f"Noise injection completed:")
        self.logger.info(f"  Noise level: {noise_level*100:.0f}%")
        self.logger.info(f"  Noise std: {noise_std:.6f}")
        self.logger.info(f"  Noise seed: {noise_seed}")
        
        X_internal_train, X_internal_val, y_internal_train, y_internal_val, split_info = self.split_internal_train_val(
            X_train_noisy, y_train_noisy
        )
        
        pipeline, cleaning_info = self.run_m4_pipeline(
            X_internal_train, y_internal_train,
            X_internal_val, y_internal_val
        )
        
        self.logger.info("Evaluating M4 pipeline on clean global sets...")
        result_global = self.evaluate_on_global_sets(
            pipeline,
            X_train_noisy, y_train_noisy,
            X_val_clean, y_val_clean,
            X_test, y_test
        )
        
        self.logger.info(f"Global evaluation completed:")
        self.logger.info(f"  Train R² (noisy): {result_global['metrics']['train_r2']:.6f}")
        self.logger.info(f"  Val R² (clean): {result_global['metrics']['val_r2']:.6f}")
        self.logger.info(f"  Test R² (clean): {result_global['metrics']['test_r2']:.6f}")
        
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            self.logger.info("")
            self.logger.info("Step 4: Physics Evaluation")
            self.run_physical_evaluation(pipeline, X_train_noisy)
        else:
            self.logger.warning("")
            self.logger.warning("="*70)
            self.logger.warning("Step 4: Physics Evaluation SKIPPED")
            self.logger.warning("Boundary models were not loaded successfully")
            self.logger.warning("="*70)
        
        if self.config.save_predictions:
            self.logger.info("Saving predictions...")
            self.save_predictions_to_excel(result_global, X_train_noisy, X_val_clean, X_test)
        
        if self.config.save_metrics:
            self.logger.info("Saving metrics...")
            self.save_metrics_to_excel(result_global)
        
        save_config(
            result_global['metrics'],
            self.config,
            self.output_dir,
            noise_info,
            split_info,
            cleaning_info
        )
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Experiment completed in {timedelta(seconds=int(elapsed_time))}")
        
        return_dict = {
            'noise_level': noise_level,
            'noise_idx': noise_idx,
            'noise_seed': noise_seed,
            'noise_std': noise_std,
            'n_train': len(X_train_noisy),
            'n_internal_train': split_info['n_internal_train'],
            'n_internal_val': split_info['n_internal_val'],
            'n_cleaned': cleaning_info.get('n_cleaned', 'N/A'),
            'n_outliers': cleaning_info.get('n_outliers', 'N/A'),
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

class M4NoiseRobustnessExperimentManager:
    """
    Manager for complete M4 pipeline noise robustness study.
    
    Coordinates multiple experiments across different noise levels:
    - Loads fixed train/val/test sets
    - Loads boundary models for LDPC and physics evaluation
    - Runs experiments for each noise level with multiple repeats
    - Aggregates and saves results
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        train_set: Training set data (clean).
        val_set: Validation set data (clean).
        test_set: Test set data (clean).
        input_boundary_model: Input boundary model.
        output_boundary_model: Output boundary model.
        all_results: Dictionary storing all experiment results.
    """
    
    def __init__(self, config: M4NoiseRobustnessConfig) -> None:
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
        """Load validation set data."""
        val_path = self.config.data_dir / self.config.validation_file
        return load_ternary_data(val_path)
    
    def _load_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test set data."""
        test_path = self.config.data_dir / self.config.test_file
        return load_ternary_data(test_path)
    
    def _load_boundary_models(self) -> None:
        """Load boundary models for LDPC and physics evaluation."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Loading Boundary Models for LDPC and Physics Evaluation")
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
            self.logger.info("All boundary models loaded - LDPC and Physics evaluation ENABLED")
            self.logger.info("="*70)
            
        except Exception as e:
            self.logger.error("")
            self.logger.error("="*70)
            self.logger.error("FAILED to load boundary models")
            self.logger.error("="*70)
            self.logger.error(f"Error: {e}")
            self.logger.error("")
            self.logger.error("LDPC correction and physics evaluation will be SKIPPED")
            self.logger.error("")
            self.logger.error("To enable LDPC and physics evaluation, ensure boundary models exist at:")
            self.logger.error(f"  1. {input_model_path}")
            self.logger.error(f"  2. {output_model_path}")
            self.logger.error("="*70)
            
            self.input_boundary_model = None
            self.output_boundary_model = None
    
    def run_all_experiments(self) -> None:
        """Run all M4 noise robustness experiments."""
        total_experiments = len(self.config.noise_levels) * self.config.n_noise_repeats
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Starting M4 Pipeline Noise Robustness Study")
        self.logger.info("="*70)
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        self.logger.info(f"Repeats per level: {self.config.n_noise_repeats}")
        self.logger.info(f"M4 Pipeline: TKMeansLOF + IADAF + LDPC")
        self.logger.info(f"Data cleaning: {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        self.logger.info(f"Synthetic samples: {self.config.n_samples_to_generate}")
        
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
                
                runner = SingleM4NoiseExperimentRunner(
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
        self.logger.info("M4 Noise Robustness Study completed successfully")
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
            
            n_train_values = [r['n_train'] for r in results]
            row['n_train'] = int(np.mean(n_train_values))
            
            n_cleaned_values = [r.get('n_cleaned', 0) for r in results if r.get('n_cleaned') != 'N/A']
            if len(n_cleaned_values) > 0:
                row['n_cleaned_mean'] = np.mean(n_cleaned_values)
                row['n_outliers_mean'] = row['n_train'] * 0.8 - row['n_cleaned_mean']
            
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
        report.append("M4 Pipeline Noise Robustness Study - Summary Report")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Noise levels: {[f'{int(n*100)}%' for n in self.config.noise_levels]}")
        report.append(f"Repeats per level: {self.config.n_noise_repeats}")
        report.append(f"Total experiments: {len(self.config.noise_levels) * self.config.n_noise_repeats}")
        
        report.append("\n" + "-"*70)
        report.append("M4 Pipeline Components")
        report.append("-"*70)
        report.append(f"Module 1: TKMeansLOF (Data Cleaning) - {'Enabled' if self.config.enable_cleaning else 'Disabled'}")
        report.append(f"Module 2: IADAF (Data Augmentation) - {self.config.n_samples_to_generate} synthetic samples")
        report.append(f"Module 3: LDPC (Boundary Correction) - ConformingCorrectedModel")
        
        report.append("\n" + "-"*70)
        report.append("Experimental Design")
        report.append("-"*70)
        report.append(f"Method: Gaussian noise injection with internal split")
        report.append(f"Noise type: Additive Gaussian (training labels only)")
        report.append(f"Internal validation ratio: {self.config.internal_val_ratio}")
        report.append(f"Training seed: {self.config.training_seed} (fixed)")
        report.append(f"Evaluation: M4 pipeline output on clean global val/test sets")
        
        report.append("\n" + "-"*70)
        report.append("Dataset Information")
        report.append("-"*70)
        report.append(f"Training: {len(self.train_set[0])} samples")
        report.append(f"Validation: {len(self.val_set[0])} samples (clean)")
        report.append(f"Test: {len(self.test_set[0])} samples (clean)")
        
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
    Main entry point for M4 pipeline noise robustness experiment.
    
    Configures and runs complete noise robustness study with M4 pipeline.
    """
    config = M4NoiseRobustnessConfig()
    
    config.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    config.n_noise_repeats = 10
    config.training_seed = 42
    config.noise_base_seed = 20000
    config.internal_val_ratio = 0.2
    config.internal_split_seed = 999
    
    config.enable_cleaning = True
    config.n_samples_to_generate = 2000
    
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
    logger.info("M4 Pipeline Noise Robustness Experiment")
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
    logger.info("M4 Pipeline components:")
    logger.info(f"  - TKMeansLOF cleaning: {'Enabled' if config.enable_cleaning else 'Disabled'}")
    logger.info(f"  - IADAF augmentation: {config.n_samples_to_generate} samples")
    logger.info(f"  - LDPC boundary correction: Enabled")
    logger.info("")
    logger.info("Key features:")
    logger.info(f"  - Internal split: {config.internal_val_ratio*100:.0f}% validation")
    logger.info(f"  - Training seed: {config.training_seed} (fixed for reproducibility)")
    logger.info(f"  - Noise base seed: {config.noise_base_seed}")
    logger.info(f"  - Physics evaluation: boundary_lambda={config.boundary_decay_lambda}, smoothness_lambda={config.smoothness_decay_lambda}")
    
    manager = M4NoiseRobustnessExperimentManager(config)
    manager.run_all_experiments()
    
    logger.info("")
    logger.info("M4 pipeline noise robustness experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()