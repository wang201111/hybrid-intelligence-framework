"""
================================================================================
Baseline Ablation Experiment - Solubility System
================================================================================

Evaluates baseline DNN performance using k-fold cross-validation without any
data augmentation, outlier detection, or physical constraints.

Features:
    - K-fold cross-validation for robust performance estimation
    - TSTR (Train on Synthetic, Test on Real) evaluation framework
    - Physical consistency evaluation using boundary models
    - Complete metrics tracking and reporting
    - No visualization code (pure evaluation)

Standardization:
    - PEP 8 compliant
    - Complete type hints (PEP 484)
    - Google-style docstrings (PEP 257)
    - Logging module instead of print
    - English documentation throughout
    - Relative paths with configurable base directory
    - No hardcoded paths

================================================================================
"""

# Standard library imports
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src directory to path for imports
# File location: experiments/solubility/ablation/baseline_experiment.py
# Need 4 parents to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from utils_solubility import (
    DNN,
    PhysicalConsistencyEvaluator,
    PhysicsConfig,
    TSTREvaluator,
)
from binary_predictor import BinaryPredictor

warnings.filterwarnings('ignore')


# ==============================================================================
# Data Loading Function (Fixed for .xlsx files)
# ==============================================================================

def load_ternary_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ternary system data from Excel file.
    
    Fixed version that correctly handles .xlsx files using pd.read_excel
    instead of pd.read_csv to avoid encoding issues.
    
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
    
    # Read Excel file
    data = pd.read_excel(filepath, engine='openpyxl')
    
    # Expected columns: Temperature (T), Input Component (W_input), Output Component (W_output)
    # Column names may vary, so we check common patterns
    col_names = data.columns.tolist()
    
    # Find temperature column
    temp_col = None
    for col in col_names:
        if 'T' in col or 'temp' in col.lower():
            temp_col = col
            break
    
    if temp_col is None:
        # Assume first column is temperature
        temp_col = col_names[0]
    
    # Remaining columns are composition data
    comp_cols = [col for col in col_names if col != temp_col]
    
    if len(comp_cols) < 2:
        raise ValueError(f"Expected at least 2 composition columns, found {len(comp_cols)}")
    
    # X = [Temperature, Input_Component]
    # y = Output_Component
    X = data[[temp_col, comp_cols[0]]].values.astype(np.float32)
    y = data[comp_cols[1]].values.astype(np.float32)
    
    return X, y


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
# Configuration
# ==============================================================================

@dataclass
class BaselineConfig:
    """
    Configuration for baseline ablation experiment.
    
    Attributes:
        Data Paths:
            data_dir: Base data directory.
            ternary_data_file: Ternary system training data filename.
            test_data_file: Test data filename.
        
        Model Paths:
            models_dir: Directory containing boundary models.
            input_boundary_model_file: Input boundary model filename.
            output_boundary_model_file: Output boundary model filename.
        
        Training Parameters:
            dnn_epochs: Number of training epochs.
            dnn_learning_rate: Learning rate.
            dnn_layer_dim: Number of DNN layers.
            dnn_node_dim: Number of nodes per layer.
        
        Cross-Validation:
            k_folds: Number of folds for cross-validation.
            kfold_random_state: Random seed for fold splitting.
        
        Output Settings:
            output_dir: Output directory for results.
            save_predictions: Whether to save predictions to Excel.
            save_metrics: Whether to save metrics to Excel.
            excel_prefix: Prefix for Excel output files.
        
        Device:
            device: Computing device ('auto', 'cuda', 'cpu').
            log_level: Logging level.
    """
    # Data paths (configurable, relative to project root)
    data_dir: Path = Path('./data/solubility/split_by_temperature')
    ternary_data_file: str = 'KCl_MgCl2_H2O_low_temp.xlsx'
    test_data_file: str = 'KCl_MgCl2_H2O_high_temp.xlsx'
    
    # Model paths
    models_dir: Path = Path('./models/solubility/binary')
    input_boundary_model_file: str = 'KCl_H2O.pth'
    output_boundary_model_file: str = 'MgCl2_H2O.pth'
    
    # Binary model data paths
    binary_data_dir: Path = Path('./data/solubility/raw/binary')
    input_boundary_data_file: str = 'KCl_H2O.xlsx'
    output_boundary_data_file: str = 'MgCl2_H2O.xlsx'
    
    # Training parameters
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    # Cross-validation parameters
    k_folds: int = 5
    kfold_random_state: int = 42
    
    # Output settings
    output_dir: Path = Path('./results/solubility/ablation/baseline')
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'baseline_'
    
    # Device and logging
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self):
        """Post-initialization: resolve device and create full paths."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Convert relative paths to absolute paths relative to PROJECT_ROOT
        if not self.data_dir.is_absolute():
            self.data_dir = PROJECT_ROOT / self.data_dir
        if not self.models_dir.is_absolute():
            self.models_dir = PROJECT_ROOT / self.models_dir
        if not self.binary_data_dir.is_absolute():
            self.binary_data_dir = PROJECT_ROOT / self.binary_data_dir
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir
        
        # Construct full paths
        self.ternary_data_path = self.data_dir / self.ternary_data_file
        self.test_data_path = self.data_dir / self.test_data_file
        self.input_boundary_model_path = self.models_dir / self.input_boundary_model_file
        self.output_boundary_model_path = self.models_dir / self.output_boundary_model_file
        self.input_boundary_data_path = self.binary_data_dir / self.input_boundary_data_file
        self.output_boundary_data_path = self.binary_data_dir / self.output_boundary_data_file


# ==============================================================================
# Utility Functions
# ==============================================================================

def create_k_folds(
    X: np.ndarray, 
    y: np.ndarray, 
    n_splits: int = 5, 
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        X: Input features array of shape (n_samples, n_features).
        y: Target values array of shape (n_samples,).
        n_splits: Number of folds for cross-validation.
        random_state: Random seed for reproducibility.
    
    Returns:
        List of dictionaries, each containing:
            - fold_idx: Fold index (0 to n_splits-1).
            - X_train: Training features.
            - y_train: Training targets.
            - X_val: Validation features.
            - y_val: Validation targets.
            - train_indices: Training sample indices.
            - val_indices: Validation sample indices.
    
    Examples:
        >>> X = np.random.rand(100, 2)
        >>> y = np.random.rand(100)
        >>> folds = create_k_folds(X, y, n_splits=5)
        >>> len(folds)
        5
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_data = {
            'fold_idx': fold_idx,
            'X_train': X[train_idx],
            'y_train': y[train_idx],
            'X_val': X[val_idx],
            'y_val': y[val_idx],
            'train_indices': train_idx,
            'val_indices': val_idx
        }
        folds.append(fold_data)
    
    return folds


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
        ...     metrics_dict={'metrics': {'train_r2': 0.95, ...}},
        ...     physics_results={'total_score': 0.87, ...},
        ...     save_path=Path('./results/fold_0_metrics.xlsx')
        ... )
    """
    if logger_instance is None:
        logger_instance = logger
    
    data = []
    
    # Extract inner metrics (handle nested structure)
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
        # Physics total score (key is 'overall_score' not 'total_score')
        physics_total = physics_results.get('overall_score', float('nan'))
        data.append(['Physics Score', physics_total])
        
        # Boundary consistency (direct key 'boundary_score')
        physics_boundary = physics_results.get('boundary_score', float('nan'))
        data.append(['Boundary Consistency', physics_boundary])
        
        # Thermodynamic smoothness (direct key 'smoothness_score')
        physics_smoothness = physics_results.get('smoothness_score', float('nan'))
        data.append(['Thermodynamic Smoothness', physics_smoothness])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    
    # Save to Excel
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    logger_instance.debug(f"Metrics saved: {save_path.name}")


def save_experiment_config(
    config: BaselineConfig,
    fold_idx: Optional[int],
    save_dir: Path
) -> None:
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Experiment configuration.
        fold_idx: Fold index (None for summary).
        save_dir: Directory to save configuration.
    """
    config_dict = {
        'experiment_info': {
            'experiment_name': 'Baseline Ablation Study',
            'experiment_type': 'K-Fold Cross-Validation',
            'version': '3.0 (Standardized)',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device,
            'fold_idx': fold_idx
        },
        'data_paths': {
            'ternary_data': str(config.ternary_data_path),
            'test_data': str(config.test_data_path),
            'input_boundary_model': str(config.input_boundary_model_path),
            'output_boundary_model': str(config.output_boundary_model_path)
        },
        'training_parameters': {
            'dnn_epochs': config.dnn_epochs,
            'dnn_learning_rate': config.dnn_learning_rate,
            'dnn_layer_dim': config.dnn_layer_dim,
            'dnn_node_dim': config.dnn_node_dim
        },
        'cross_validation': {
            'k_folds': config.k_folds,
            'kfold_random_state': config.kfold_random_state,
            'validation_method': 'K-Fold with fixed random seed',
            'note': 'Each fold uses different train/val split'
        },
        'output_settings': {
            'save_predictions': config.save_predictions,
            'save_metrics': config.save_metrics,
            'excel_prefix': config.excel_prefix
        }
    }
    
    config_path = save_dir / 'configs' / 'experiment_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Configuration saved: {config_path}")


# ==============================================================================
# Baseline Experiment Evaluation
# ==============================================================================

def run_baseline_with_tstr_evaluator(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    config: BaselineConfig
) -> Tuple[Dict[str, Any], nn.Module, StandardScaler, StandardScaler]:
    """
    Run baseline evaluation using TSTR evaluator.
    
    Uses TSTREvaluator to train DNN on real training data and evaluate
    on validation and test sets.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        X_test: Test features.
        y_test: Test targets.
        config: Experiment configuration.
    
    Returns:
        Tuple containing:
            - result_dict: Dictionary with metrics, history, predictions
            - dnn_model: Trained DNN model
            - x_scaler: Feature scaler
            - y_scaler: Target scaler
    """
    logger.info("="*70)
    logger.info("Baseline Evaluation - TSTR Framework")
    logger.info("="*70)
    
    # Create TSTR evaluator
    logger.info("Creating TSTR evaluator...")
    
    evaluator = TSTREvaluator(
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
        config=PhysicsConfig(
            tstr_epochs=config.dnn_epochs,
            tstr_lr=config.dnn_learning_rate,
            dnn_layer_dim=config.dnn_layer_dim,
            dnn_node_dim=config.dnn_node_dim,
            tstr_device=config.device
        )
    )
    
    logger.info("TSTR evaluator created successfully")
    
    # Evaluate baseline (train on real data)
    logger.info("Evaluating baseline (training on real data)...")
    
    result = evaluator.evaluate(
        X_syn=X_train,
        y_syn=y_train,
        epochs=config.dnn_epochs,
        verbose=True
    )
    
    # Extract results
    metrics_dict = result['metrics']
    dnn_model = result['model']
    x_scaler = result['x_scaler']
    y_scaler = result['y_scaler']
    
    # Get predictions (only val and test are returned)
    predictions = result['predictions']
    
    # Calculate training set predictions and metrics (not returned by evaluator)
    device = torch.device(config.device)
    dnn_model.eval()
    with torch.no_grad():
        X_train_scaled = x_scaler.transform(X_train)
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_pred_scaled = dnn_model(X_train_tensor).cpu().numpy()
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).flatten()
    
    # Calculate training metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # Add training metrics to metrics dict
    metrics_dict['train_r2'] = float(train_r2)
    metrics_dict['train_rmse'] = float(train_rmse)
    metrics_dict['train_mae'] = float(train_mae)
    
    # Add training predictions
    predictions['train'] = y_train_pred
    
    # Create true values dict
    true_values = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    # Prepare result dictionary (compatible with old format)
    result_dict = {
        'metrics': metrics_dict,
        'history': None,  # TSTREvaluator doesn't return training history
        'predictions': predictions,
        'true_values': true_values
    }
    
    logger.info("TSTR evaluation completed")
    
    return result_dict, dnn_model, x_scaler, y_scaler


# ==============================================================================
# Baseline Ablation Study Class
# ==============================================================================

class BaselineAblationStudy:
    """
    Baseline ablation study for solubility prediction.
    
    Evaluates DNN performance using real training data without any
    augmentation or physical constraints. Serves as the baseline
    for comparison with enhanced methods.
    
    Attributes:
        config: Experiment configuration.
        results: Dictionary storing experiment results.
        input_boundary_model: Boundary model for input component.
        output_boundary_model: Boundary model for output component.
        logger: Logger instance for this study.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize baseline ablation study.
        
        Args:
            config: Experiment configuration.
        """
        self.config = config
        self.results = {}
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        # Set random seeds for reproducibility
        np.random.seed(config.kfold_random_state)
        torch.manual_seed(config.kfold_random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.kfold_random_state)
        
        # Load boundary models
        self._load_boundary_models()
        
        self.logger.info("Baseline ablation study initialized")
    
    def _load_boundary_models(self) -> None:
        """
        Load trained boundary models.
        
        Attempts to load pre-trained boundary models. If loading fails,
        logs a warning and sets models to None (physics evaluation will be skipped).
        """
        self.logger.info("="*70)
        self.logger.info("Loading Boundary Models")
        self.logger.info("="*70)
        
        try:
            # Load input boundary model
            if self.config.input_boundary_model_path.exists():
                self.input_boundary_model = BinaryPredictor.load(
                    str(self.config.input_boundary_model_path)
                )
                self.logger.info(f"Input boundary model loaded: {self.config.input_boundary_model_file}")
            else:
                raise FileNotFoundError(
                    f"Input boundary model not found: {self.config.input_boundary_model_path}"
                )
            
            # Load output boundary model
            if self.config.output_boundary_model_path.exists():
                self.output_boundary_model = BinaryPredictor.load(
                    str(self.config.output_boundary_model_path)
                )
                self.logger.info(f"Output boundary model loaded: {self.config.output_boundary_model_file}")
            else:
                raise FileNotFoundError(
                    f"Output boundary model not found: {self.config.output_boundary_model_path}"
                )
            
            self.logger.info("Boundary models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load boundary models: {e}")
            self.logger.warning("Physics evaluation will be skipped")
            self.input_boundary_model = None
            self.output_boundary_model = None
    
    def run_experiment(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> None:
        """
        Run complete baseline experiment.
        
        Executes the full experimental pipeline:
            1. Train DNN using TSTR evaluator
            2. Run physical consistency evaluation (if models available)
            3. Save results (predictions, metrics, models)
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            X_test: Test features.
            y_test: Test targets.
        """
        self.logger.info("")
        self.logger.info("█"*70)
        self.logger.info("█" + " Baseline Ablation Experiment ".center(68) + "█")
        self.logger.info("█"*70)
        self.logger.info("")
        
        # Step 1: TSTR Evaluation
        result, dnn, x_scaler, y_scaler = run_baseline_with_tstr_evaluator(
            X_train, y_train, X_val, y_val, X_test, y_test, self.config
        )
        
        # Store results
        self.results['metrics'] = result
        self.results['model'] = dnn
        self.results['x_scaler'] = x_scaler
        self.results['y_scaler'] = y_scaler
        
        # Step 2: Save model
        model_path = self.config.output_dir / 'models' / f'{self.config.excel_prefix}model.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dnn.state_dict(), model_path)
        self.logger.info(f"Model saved: {model_path}")
        
        # Step 3: Physical evaluation (if boundary models available)
        if self.input_boundary_model is not None and self.output_boundary_model is not None:
            self._run_physical_evaluation(dnn, x_scaler, y_scaler, X_train, y_train)
        else:
            self.logger.warning("Boundary models not available, skipping physics evaluation")
        
        # Step 4: Save results
        self._save_results(result)
    
    def _run_physical_evaluation(
        self,
        dnn: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Run physical consistency evaluation.
        
        Evaluates model predictions against physical constraints using
        boundary models and thermodynamic smoothness analysis.
        
        Args:
            dnn: Trained DNN model.
            x_scaler: Feature scaler.
            y_scaler: Target scaler.
            X_train: Training features (for reference).
            y_train: Training targets (for reference).
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Physical Consistency Evaluation")
        self.logger.info("="*70)
        
        # Generate phase diagram predictions
        T_range = (-40, 200)
        W_input_range = (0, 55)
        resolution = 100
        
        T_test = np.linspace(T_range[0], T_range[1], resolution)
        W_input_test = np.linspace(W_input_range[0], W_input_range[1], resolution)
        T_mesh, W_input_mesh = np.meshgrid(T_test, W_input_test)
        X_phase = np.column_stack([T_mesh.ravel(), W_input_mesh.ravel()])
        
        # Make predictions
        device = torch.device(self.config.device)
        dnn.eval()
        X_phase_scaled = x_scaler.transform(X_phase)
        
        with torch.no_grad():
            y_phase_pred = dnn(torch.FloatTensor(X_phase_scaled).to(device)).cpu().numpy()
        
        y_phase_pred = y_scaler.inverse_transform(y_phase_pred).flatten()
        predicted_data = np.column_stack([X_phase[:, 0], X_phase[:, 1], y_phase_pred])
        
        # Create physics evaluator
        evaluator = PhysicalConsistencyEvaluator(
            input_boundary_model=self.input_boundary_model,
            output_boundary_model=self.output_boundary_model,
            boundary_decay_lambda=5.0,
            smoothness_decay_lambda=15.0
        )
        
        # Run evaluation
        physics_score, physics_results = evaluator.evaluate_full(
            dnn, x_scaler, y_scaler, predicted_data, self.config.device
        )
        
        self.logger.info(f"Physics Score: {physics_score:.6f}")
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(physics_results)
        report_path = self.config.output_dir / 'data' / 'physical_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Physics evaluation report saved: {report_path}")
        
        # Store results
        self.results['physical_evaluation'] = physics_results
        self.results['physics_score'] = physics_score
        self.results['predicted_data'] = predicted_data
    
    def _save_results(self, result: Dict[str, Any]) -> None:
        """
        Save experiment results to files.
        
        Saves predictions and metrics to Excel files if configured.
        
        Args:
            result: Dictionary containing experiment results.
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Saving Results")
        self.logger.info("="*70)
        
        # Save predictions to Excel
        if self.config.save_predictions:
            self.logger.info("Saving predictions to Excel...")
            excel_dir = self.config.output_dir / 'excel'
            excel_dir.mkdir(parents=True, exist_ok=True)
            
            # Get predictions and true values
            predictions = result['predictions']
            true_values = result['true_values']
            
            # Training set
            df_train = pd.DataFrame({
                'y_true': true_values['train'].flatten(),
                'y_pred': predictions['train'].flatten(),
                'residual': true_values['train'].flatten() - predictions['train'].flatten()
            })
            train_path = excel_dir / f'{self.config.excel_prefix}train_predictions.xlsx'
            df_train.to_excel(train_path, index=False, engine='openpyxl')
            self.logger.debug(f"Train predictions saved: {train_path.name}")
            
            # Validation set
            df_val = pd.DataFrame({
                'y_true': true_values['val'].flatten(),
                'y_pred': predictions['val'].flatten(),
                'residual': true_values['val'].flatten() - predictions['val'].flatten()
            })
            val_path = excel_dir / f'{self.config.excel_prefix}val_predictions.xlsx'
            df_val.to_excel(val_path, index=False, engine='openpyxl')
            self.logger.debug(f"Val predictions saved: {val_path.name}")
            
            # Test set
            df_test = pd.DataFrame({
                'y_true': true_values['test'].flatten(),
                'y_pred': predictions['test'].flatten(),
                'residual': true_values['test'].flatten() - predictions['test'].flatten()
            })
            test_path = excel_dir / f'{self.config.excel_prefix}test_predictions.xlsx'
            df_test.to_excel(test_path, index=False, engine='openpyxl')
            self.logger.debug(f"Test predictions saved: {test_path.name}")
        
        # Save metrics to Excel
        if self.config.save_metrics:
            self.logger.info("Saving metrics to Excel...")
            excel_dir = self.config.output_dir / 'excel'
            excel_dir.mkdir(parents=True, exist_ok=True)
            
            physics_results = self.results.get('physical_evaluation', None)
            
            save_fold_metrics_to_excel(
                metrics_dict=result,
                physics_results=physics_results,
                save_path=excel_dir / f'{self.config.excel_prefix}metrics.xlsx',
                logger_instance=self.logger
            )
            
            self.logger.info(f"Metrics saved: {self.config.excel_prefix}metrics.xlsx")
    
    def print_summary(self) -> None:
        """Print experiment summary."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Experiment Summary")
        self.logger.info("="*70)
        
        metrics = self.results['metrics']
        inner_metrics = metrics['metrics']
        
        # Print metrics table
        self.logger.info("")
        self.logger.info(f"{'Metric':<25} {'Train':<15} {'Val':<15} {'Test':<15}")
        self.logger.info("-"*70)
        
        metric_mapping = {
            'R²': ('train_r2', 'val_r2', 'test_r2'),
            'RMSE': ('train_rmse', 'val_rmse', 'test_rmse'),
            'MAE': ('train_mae', 'val_mae', 'test_mae')
        }
        
        for metric_name, (train_key, val_key, test_key) in metric_mapping.items():
            train_val = inner_metrics.get(train_key, float('nan'))
            val_val = inner_metrics.get(val_key, float('nan'))
            test_val = inner_metrics.get(test_key, float('nan'))
            self.logger.info(
                f"{metric_name:<25} {train_val:<15.6f} {val_val:<15.6f} {test_val:<15.6f}"
            )
        
        self.logger.info("")
        
        # Physics score
        if 'physical_evaluation' in self.results:
            physics_score = self.results['physical_evaluation']['overall_score']
            self.logger.info(f"Physics Score: {physics_score:.6f}")
            self.logger.info("")
        
        # Output directories
        self.logger.info("Output Files:")
        self.logger.info(f"  Configs: {self.config.output_dir / 'configs'}")
        self.logger.info(f"  Data: {self.config.output_dir / 'data'}")
        self.logger.info(f"  Excel: {self.config.output_dir / 'excel'}")
        self.logger.info(f"  Models: {self.config.output_dir / 'models'}")
        self.logger.info("")
        self.logger.info("="*70)


# ==============================================================================
# K-Fold Experiment Manager
# ==============================================================================

class KFoldExperimentManager:
    """
    Manager for K-fold cross-validation experiments.
    
    Coordinates running multiple folds, aggregating results, and
    generating summary statistics.
    
    Attributes:
        base_config: Base configuration for experiments.
        n_folds: Number of folds for cross-validation.
        all_results: List of results from all folds.
        main_output_dir: Main output directory for all folds.
        logger: Logger instance for this manager.
    """
    
    def __init__(self, base_config: BaselineConfig, n_folds: int = 5):
        """
        Initialize K-fold experiment manager.
        
        Args:
            base_config: Base configuration for experiments.
            n_folds: Number of folds for cross-validation.
        """
        self.base_config = base_config
        self.n_folds = n_folds
        self.all_results = []
        self.logger = get_logger(self.__class__.__name__, base_config.log_level)
        
        # Create main output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.main_output_dir = Path(f'./baseline_kfold_results_{timestamp}')
        self.main_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Main output directory: {self.main_output_dir}")
    
    def run_all_folds(self) -> None:
        """
        Run experiments for all K folds.
        
        Executes the complete experimental pipeline:
            1. Load data and create K-fold splits
            2. Run experiment for each fold
            3. Aggregate results across folds
            4. Generate summary statistics and reports
        """
        self.logger.info("")
        self.logger.info("█"*70)
        self.logger.info("█" + f" K-Fold Cross-Validation ({self.n_folds} folds) ".center(68) + "█")
        self.logger.info("█"*70)
        
        total_start = time.time()
        
        # Load data and create K-fold splits
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Loading Data and Creating K-Fold Splits")
        self.logger.info("="*70)
        
        X_ternary, y_ternary = load_ternary_data(self.base_config.ternary_data_path)
        X_test, y_test = load_ternary_data(self.base_config.test_data_path)
        
        self.logger.info(f"Data loaded successfully")
        self.logger.info(f"  Training data: {len(X_ternary)} samples")
        self.logger.info(f"  Test data: {len(X_test)} samples")
        
        # Create K-fold splits
        folds = create_k_folds(
            X_ternary, 
            y_ternary, 
            n_splits=self.n_folds, 
            random_state=self.base_config.kfold_random_state
        )
        
        self.logger.info("")
        self.logger.info(
            f"K-fold splits created (n_splits={self.n_folds}, "
            f"random_state={self.base_config.kfold_random_state})"
        )
        for i, fold in enumerate(folds):
            self.logger.info(
                f"  Fold {i}: Train={len(fold['X_train'])}, Val={len(fold['X_val'])}"
            )
        
        # Run each fold
        for fold_idx, fold_data in enumerate(folds):
            fold_start = time.time()
            
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Starting Fold {fold_idx}")
            self.logger.info("="*70)
            
            # Create fold-specific output directory
            fold_dir = self.main_output_dir / f'fold_{fold_idx}'
            self._create_fold_directories(fold_dir)
            
            # Create fold-specific configuration
            fold_config = BaselineConfig()
            fold_config.__dict__.update(self.base_config.__dict__)
            fold_config.output_dir = fold_dir
            
            # Extract fold data
            X_train = fold_data['X_train']
            y_train = fold_data['y_train']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            
            # Run experiment
            study = BaselineAblationStudy(fold_config)
            study.run_experiment(X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Store results
            fold_result = {
                'fold_idx': fold_idx,
                'config': fold_config,
                'metrics': study.results['metrics'],
                'physics_score': study.results.get('physics_score', None),
                'physics_boundary': None,
                'physics_smoothness': None
            }
            
            # Extract physics scores (use correct keys from evaluate_full)
            if 'physical_evaluation' in study.results:
                phys_eval = study.results['physical_evaluation']
                fold_result['physics_score'] = phys_eval.get('overall_score', None)
                fold_result['physics_boundary'] = phys_eval.get('boundary_score', None)
                fold_result['physics_smoothness'] = phys_eval.get('smoothness_score', None)
            
            self.all_results.append(fold_result)
            
            # Save fold configuration
            save_experiment_config(fold_config, fold_idx, fold_dir)
            
            fold_elapsed = time.time() - fold_start
            self.logger.info(f"Fold {fold_idx} completed in {timedelta(seconds=int(fold_elapsed))}")
        
        # Generate summary
        total_elapsed = time.time() - total_start
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("All Folds Completed")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {timedelta(seconds=int(total_elapsed))}")
        self.logger.info(f"Average time per fold: {timedelta(seconds=int(total_elapsed/self.n_folds))}")
        
        # Generate summary outputs
        self._generate_summary()
    
    def _create_fold_directories(self, fold_dir: Path) -> None:
        """
        Create directory structure for a fold.
        
        Args:
            fold_dir: Base directory for the fold.
        """
        (fold_dir / 'configs').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'data').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'excel').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'models').mkdir(parents=True, exist_ok=True)
    
    def _generate_summary(self) -> None:
        """
        Generate summary statistics and reports across all folds.
        
        Creates summary directory with:
            - Aggregated metrics
            - Summary predictions
            - Text report
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Generating Summary Statistics")
        self.logger.info("="*70)
        
        summary_dir = self.main_output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        # Save summary metrics
        self._save_summary_metrics(summary_dir)
        
        # Save summary predictions
        self._save_summary_predictions(summary_dir)
        
        # Generate text report
        self._generate_text_report(summary_dir)
        
        self.logger.info("Summary generation completed")
    
    def _save_summary_metrics(self, summary_dir: Path) -> None:
        """
        Save aggregated metrics across all folds.
        
        Args:
            summary_dir: Directory to save summary metrics.
        """
        self.logger.info("Saving summary metrics...")
        
        # Collect metrics from all folds
        metric_keys = ['train_r2', 'train_rmse', 'train_mae',
                      'val_r2', 'val_rmse', 'val_mae',
                      'test_r2', 'test_rmse', 'test_mae']
        
        metric_names = ['Train R²', 'Train RMSE', 'Train MAE',
                       'Val R²', 'Val RMSE', 'Val MAE',
                       'Test R²', 'Test RMSE', 'Test MAE']
        
        data = []
        
        for metric_name, metric_key in zip(metric_names, metric_keys):
            values = [r['metrics']['metrics'][metric_key] for r in self.all_results]
            mean_val = np.mean(values)
            
            data.append([metric_name, mean_val])
        
        # Add physics metrics
        physics_score_values = [
            r.get('physics_score', float('nan')) 
            for r in self.all_results
        ]
        physics_score_values = [v for v in physics_score_values if v is not None and not np.isnan(v)]
        
        if len(physics_score_values) > 0:
            data.append(['Physics Score', np.mean(physics_score_values)])
        
        physics_boundary_values = [
            r.get('physics_boundary', float('nan')) 
            for r in self.all_results
        ]
        physics_boundary_values = [v for v in physics_boundary_values if v is not None and not np.isnan(v)]
        
        if len(physics_boundary_values) > 0:
            data.append(['Boundary Consistency', np.mean(physics_boundary_values)])
        
        physics_smoothness_values = [
            r.get('physics_smoothness', float('nan')) 
            for r in self.all_results
        ]
        physics_smoothness_values = [v for v in physics_smoothness_values if v is not None and not np.isnan(v)]
        
        if len(physics_smoothness_values) > 0:
            data.append(['Thermodynamic Smoothness', np.mean(physics_smoothness_values)])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['Metric', 'Mean Value'])
        
        # Save to Excel
        summary_path = summary_dir / 'summary_metrics.xlsx'
        df.to_excel(summary_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Summary metrics saved: {summary_path.name}")
    
    def _save_summary_predictions(self, summary_dir: Path) -> None:
        """
        Save aggregated predictions across all folds.
        
        Args:
            summary_dir: Directory to save summary predictions.
        """
        self.logger.info("Saving summary predictions...")
        
        # Get test set predictions from all folds
        y_test = self.all_results[0]['metrics']['true_values']['test'].flatten()
        
        data = {'y_true': y_test}
        
        # Add predictions from each fold
        for result in self.all_results:
            fold_idx = result['fold_idx']
            y_pred = result['metrics']['predictions']['test'].flatten()
            data[f'y_pred_fold{fold_idx}'] = y_pred
        
        # Calculate statistics
        y_test_preds = np.array([
            r['metrics']['predictions']['test'].flatten() 
            for r in self.all_results
        ])
        data['y_pred_mean'] = np.mean(y_test_preds, axis=0)
        data['y_pred_std'] = np.std(y_test_preds, axis=0, ddof=1)
        data['residual_mean'] = y_test - data['y_pred_mean']
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to Excel
        detail_path = summary_dir / 'test_predictions_summary.xlsx'
        df.to_excel(detail_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Summary predictions saved: {detail_path.name}")
    
    def _generate_text_report(self, summary_dir: Path) -> None:
        """
        Generate comprehensive text report.
        
        Args:
            summary_dir: Directory to save text report.
        """
        self.logger.info("Generating text report...")
        
        report = []
        report.append("="*70)
        report.append("Baseline Ablation Study - K-Fold Cross-Validation Summary")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"K-Folds: {self.n_folds}")
        report.append(f"Random Seed: {self.base_config.kfold_random_state}")
        report.append(f"Device: {self.base_config.device}")
        
        # Configuration
        report.append("\n" + "-"*70)
        report.append("Configuration")
        report.append("-"*70)
        report.append(f"DNN Layers: {self.base_config.dnn_layer_dim}")
        report.append(f"DNN Nodes: {self.base_config.dnn_node_dim}")
        report.append(f"Training Epochs: {self.base_config.dnn_epochs}")
        report.append(f"Learning Rate: {self.base_config.dnn_learning_rate}")
        
        # Summary statistics
        report.append("\n" + "-"*70)
        report.append("Summary Statistics")
        report.append("-"*70)
        
        metric_keys = ['train_r2', 'train_rmse', 'train_mae',
                      'val_r2', 'val_rmse', 'val_mae',
                      'test_r2', 'test_rmse', 'test_mae']
        
        metric_names = ['Train R²', 'Train RMSE', 'Train MAE',
                       'Val R²', 'Val RMSE', 'Val MAE',
                       'Test R²', 'Test RMSE', 'Test MAE']
        
        for key, name in zip(metric_keys, metric_names):
            values = [r['metrics']['metrics'][key] for r in self.all_results]
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            report.append(f"{name:20s}: {mean_val:.6f} ± {std_val:.6f}")
        
        # Physics metrics
        report.append("\nPhysics Evaluation Metrics:")
        physics_metrics = [
            ('physics_score', 'Physics Score'),
            ('physics_boundary', 'Boundary Consistency'),
            ('physics_smoothness', 'Thermodynamic Smoothness')
        ]
        
        for key, name in physics_metrics:
            values = [r.get(key, float('nan')) for r in self.all_results]
            values = [v for v in values if v is not None and not np.isnan(v)]
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                report.append(f"{name:20s}: {mean_val:.6f} ± {std_val:.6f}")
            else:
                report.append(f"{name:20s}: N/A")
        
        # Best model
        report.append("\n" + "-"*70)
        report.append("Best Model")
        report.append("-"*70)
        
        test_r2_values = [r['metrics']['metrics']['test_r2'] for r in self.all_results]
        best_idx = np.argmax(test_r2_values)
        best_fold = self.all_results[best_idx]['fold_idx']
        best_r2 = test_r2_values[best_idx]
        
        report.append(f"Best Fold (by Test R²): Fold {best_fold}")
        report.append(f"Test R²: {best_r2:.6f}")
        
        # Stability
        report.append("\n" + "-"*70)
        report.append("Stability Analysis")
        report.append("-"*70)
        
        test_r2_std = np.std(test_r2_values, ddof=1)
        test_r2_cv = (test_r2_std / np.mean(test_r2_values)) * 100
        
        report.append(f"Test R² Std Dev: {test_r2_std:.6f}")
        report.append(f"Test R² CV: {test_r2_cv:.2f}%")
        
        report.append("\n" + "="*70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = summary_dir / 'summary_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Text report saved: {report_path.name}")
        self.logger.info("\n" + report_text)


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """
    Main entry point for baseline ablation experiment.
    
    Configures and runs K-fold cross-validation baseline experiment.
    """
    # Create configuration
    config = BaselineConfig()
    
    # Log project structure
    logger.info("")
    logger.info("="*70)
    logger.info("Project Structure Information")
    logger.info("="*70)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Source Directory: {SRC_DIR}")
    logger.info(f"Data Directory: {config.data_dir}")
    logger.info(f"Models Directory: {config.models_dir}")
    logger.info(f"Output Directory: {config.output_dir}")
    
    # Configure experiment
    config.k_folds = 5
    config.kfold_random_state = 2
    config.dnn_epochs = 1000
    config.dnn_learning_rate = 0.00831
    config.dnn_layer_dim = 4
    config.dnn_node_dim = 128
    config.save_predictions = True
    config.save_metrics = True
    config.log_level = logging.INFO
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Baseline Ablation Experiment v3.0 (Standardized)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-Folds: {config.k_folds}")
    logger.info(f"Random Seed: {config.kfold_random_state}")
    logger.info(f"Evaluation Method: TSTR (Train on Synthetic, Test on Real)")
    logger.info("")
    logger.info("Features:")
    logger.info("  - K-fold cross-validation for robust evaluation")
    logger.info("  - Physical consistency evaluation using boundary models")
    logger.info("  - Complete metrics tracking and reporting")
    logger.info("  - No data augmentation (baseline)")
    logger.info("  - No outlier detection (baseline)")
    logger.info("  - No physical constraints (baseline)")
    
    # Run K-fold experiment
    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("Baseline experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()