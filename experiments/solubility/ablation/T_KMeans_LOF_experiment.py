"""
================================================================================
T-KMeans-LOF Ablation Study - K-Fold Cross-Validation Framework
================================================================================

Evaluates DNN performance with TKMeansLOF data cleaning using k-fold 
cross-validation.

Pipeline:
    1. Data Loading → Load ternary system training and test data
    2. TKMeansLOF Cleaning → Remove outliers from training data
    3. K-Fold Split → Create folds from cleaned data
    4. Model Training → Train DNN on each fold
    5. Evaluation → Evaluate on validation and test sets

Features:
    - K-fold cross-validation for robust evaluation
    - TKMeansLOF cleaning (Temperature-based K-Means + LOF)
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
    - ALL HYPERPARAMETERS KEPT IDENTICAL TO ORIGINAL

================================================================================
"""

# Standard library imports
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src directory to path for imports
# File location: experiments/solubility/ablation/cleaning_experiment.py
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

# Local imports
from t_kmeans_lof import TKMeansLOF, OutlierDetectionConfig
from utils_solubility import (
    DNN,
    PhysicalConsistencyEvaluator,
    PhysicsConfig,
    TSTREvaluator,
)
from binary_predictor import BinaryPredictor

warnings.filterwarnings('ignore')


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
            - X: Feature array [n_samples, 2] - [Temperature, Input_Component]
            - y: Target array [n_samples] - Output_Component
    
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = pd.read_excel(filepath, engine='openpyxl')
    
    # Find temperature column
    col_names = data.columns.tolist()
    temp_col = None
    for col in col_names:
        if 'T' in col or 'temp' in col.lower():
            temp_col = col
            break
    
    if temp_col is None:
        temp_col = col_names[0]
    
    # Remaining columns are composition data
    comp_cols = [col for col in col_names if col != temp_col]
    
    if len(comp_cols) < 2:
        raise ValueError(f"Expected at least 2 composition columns, found {len(comp_cols)}")
    
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
class CleaningConfig:
    """
    Configuration for data cleaning ablation experiment.
    
    Attributes:
        Data Paths:
            data_dir: Base data directory.
            ternary_data_file: Ternary system training data filename.
            test_data_file: Test data filename.
        
        Model Paths:
            models_dir: Directory containing boundary models.
            input_boundary_model_file: Input boundary model filename.
            output_boundary_model_file: Output boundary model filename.
        
        Binary Model Data Paths:
            binary_data_dir: Directory containing binary system data.
            input_boundary_data_file: Input boundary data filename.
            output_boundary_data_file: Output boundary data filename.
        
        Data Cleaning:
            enable_cleaning: Whether to enable TKMeansLOF cleaning.
            cleaning_config: TKMeansLOF configuration dictionary.
                contamination: Expected proportion of outliers [0.0, 0.5] (default: 0.20).
                n_neighbors: Number of neighbors for LOF (default: 5).
                min_cluster_size: Minimum cluster size (default: 10).
                auto_select_k: Auto select optimal clusters (default: True).
                min_n_clusters: Minimum number of clusters (default: 2).
                max_n_clusters: Maximum number of clusters (default: 24).
                fixed_n_clusters: Fixed cluster count if auto_select_k=False (optional).
                random_state: Random seed (default: 42).
                enable_residual_detection: Enable moving median residual detection (default: True).
                moving_median_window: Window size for moving median (default: 5).
                consensus_weight: Weight for LOF in consensus scoring (default: 0.5).
        
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
            random_seed: Random seed for reproducibility.
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
    
    # Data cleaning parameters (matching t_kmeas_lof.py OutlierDetectionConfig)
    enable_cleaning: bool = True
    cleaning_config: Dict = field(default_factory=lambda: {
        'contamination': 0.20,              # Expected outlier proportion
        'n_neighbors': 5,                  # LOF neighbors
        'min_cluster_size': 10,             # Minimum cluster size
        'auto_select_k': True,              # Auto select clusters
        'min_n_clusters': 2,                # Minimum clusters
        'max_n_clusters': 24,               # Maximum clusters
        'fixed_n_clusters': None,           # Fixed clusters (optional)
        'random_state': 42,                 # Random seed
        'enable_residual_detection': True,  # Enable residual detection
        'moving_median_window': 5,          # Moving median window
        'consensus_weight': 0.5             # LOF weight in consensus
    })
    
    # Training parameters
    dnn_epochs: int = 1000
    dnn_learning_rate: float = 0.00831
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    # Cross-validation parameters
    k_folds: int = 5
    kfold_random_state: int = 42
    random_seed: int = 42
    
    # Output settings
    output_dir: Path = Path('./results/solubility/ablation/cleaning')
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'cleaning_'
    
    # Device and logging
    device: str = 'auto'
    log_level: int = logging.INFO
    
    def __post_init__(self) -> None:
        """Post-initialization: resolve device and create full paths."""
        # Configure device
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
# K-Fold Data Splitting
# ==============================================================================

def create_k_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create K-fold cross-validation data splits.
    
    Args:
        X: Feature data [n_samples, n_features].
        y: Target data [n_samples].
        n_splits: Number of folds.
        random_state: Random seed for reproducibility.
    
    Returns:
        List of dictionaries containing fold data:
            - 'fold_idx': Fold index
            - 'X_train': Training features
            - 'y_train': Training targets
            - 'X_val': Validation features
            - 'y_val': Validation targets
            - 'train_indices': Training indices
            - 'val_indices': Validation indices
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


# ==============================================================================
# Single Fold Experiment Runner
# ==============================================================================

class SingleFoldRunner:
    """
    Runs complete data cleaning experiment for a single fold.
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        device: Computing device.
        input_boundary_model: Input boundary model for physics evaluation.
        output_boundary_model: Output boundary model for physics evaluation.
    """
    
    def __init__(
        self,
        config: CleaningConfig,
        input_boundary_model: Optional[BinaryPredictor],
        output_boundary_model: Optional[BinaryPredictor]
    ) -> None:
        """
        Initialize fold runner.
        
        Args:
            config: Experiment configuration.
            input_boundary_model: Input boundary model.
            output_boundary_model: Output boundary model.
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__, level=config.log_level)
        self.device = torch.device(config.device)
        self.input_boundary_model = input_boundary_model
        self.output_boundary_model = output_boundary_model
    
    def run(
        self,
        fold_idx: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_dir: Path
    ) -> Dict[str, Any]:
        """
        Run complete experiment for one fold.
        
        Steps:
            1. Train DNN on cleaned training data
            2. Evaluate on validation and test sets
            3. Compute physics metrics
        
        Args:
            fold_idx: Index of current fold.
            X_train: Training features [N_train, 2] (already cleaned).
            y_train: Training targets [N_train] (already cleaned).
            X_val: Validation features [N_val, 2] (already cleaned).
            y_val: Validation targets [N_val] (already cleaned).
            X_test: Test features [N_test, 2].
            y_test: Test targets [N_test].
            fold_dir: Directory to save fold results.
        
        Returns:
            Dictionary containing:
                - 'fold_idx': Fold index
                - 'metrics': Complete evaluation metrics
                - 'physics_score': Physics evaluation score
        """
        self.logger.info("="*70)
        self.logger.info(f"Starting Fold {fold_idx}")
        self.logger.info("="*70)
        
        # Step 1: Train DNN using TSTREvaluator
        self.logger.info("\n[Step 1] Training DNN with TSTREvaluator")
        result = self._train_dnn_tstr(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        model = result['model']
        x_scaler = result['x_scaler']
        y_scaler = result['y_scaler']
        
        # Step 2: Compute physics metrics
        self.logger.info("\n[Step 2] Computing physics metrics")
        physics_results = self._compute_physics_metrics(
            model, x_scaler, y_scaler
        )
        
        # Save results
        self._save_fold_results(
            fold_idx, fold_dir, result, physics_results
        )
        
        # Combine results - include complete result from TSTREvaluator
        return {
            'fold_idx': fold_idx,
            'metrics': result['metrics'],  # ML metrics
            'predictions': result.get('predictions', {}),  # Predictions
            'true_values': result.get('true_values', {}),  # True values
            'physics_score': physics_results.get('physics_score', None),
            'physics_boundary': physics_results.get('physics_boundary', None),
            'physics_smoothness': physics_results.get('physics_smoothness', None)
        }
    
    def _train_dnn_tstr(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train DNN using TSTREvaluator.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary with model, scalers, and metrics.
        """
        # Create TSTREvaluator
        evaluator = TSTREvaluator(
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            config=PhysicsConfig(
                tstr_epochs=self.config.dnn_epochs,
                tstr_lr=self.config.dnn_learning_rate,
                dnn_layer_dim=self.config.dnn_layer_dim,
                dnn_node_dim=self.config.dnn_node_dim,
                tstr_device=self.config.device  # Use config.device instead of self.device
            )
        )
        
        # Train model
        result = evaluator.evaluate(
            X_syn=X_train,
            y_syn=y_train,
            epochs=self.config.dnn_epochs,
            verbose=True
        )
        
        metrics_dict = result['metrics']
        
        self.logger.info(f"Training completed")
        self.logger.info(f"  Train: RMSE={metrics_dict['train_rmse']:.4f}, R²={metrics_dict['train_r2']:.4f}")
        self.logger.info(f"  Val:   RMSE={metrics_dict['val_rmse']:.4f}, R²={metrics_dict['val_r2']:.4f}")
        self.logger.info(f"  Test:  RMSE={metrics_dict['test_rmse']:.4f}, R²={metrics_dict['test_r2']:.4f}")
        
        return result
    
    def _compute_physics_metrics(
        self,
        model: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler
    ) -> Dict[str, float]:
        """
        Compute physics consistency metrics.
        
        Args:
            model: Trained DNN model.
            x_scaler: Feature scaler.
            y_scaler: Target scaler.
        
        Returns:
            Dictionary with physics scores.
        """
        if self.input_boundary_model is None or self.output_boundary_model is None:
            self.logger.warning("Boundary models not available, skipping physics evaluation")
            return {}
        
        try:
            # Generate phase diagram predictions for smoothness evaluation
            T_range = (-40, 200)
            W_input_range = (0, 55)
            resolution = 100
            
            T_grid = np.linspace(T_range[0], T_range[1], resolution)
            W_input_grid = np.linspace(W_input_range[0], W_input_range[1], resolution)
            T_mesh, W_input_mesh = np.meshgrid(T_grid, W_input_grid)
            
            X_phase = np.column_stack([T_mesh.ravel(), W_input_mesh.ravel()])
            
            # Predict using the model
            model.eval()
            with torch.no_grad():
                X_phase_scaled = x_scaler.transform(X_phase)
                X_phase_tensor = torch.FloatTensor(X_phase_scaled).to(self.device)
                y_phase_pred_scaled = model(X_phase_tensor).cpu().numpy()
                y_phase_pred = y_scaler.inverse_transform(y_phase_pred_scaled).flatten()
            
            # Construct predicted_data [T, W_input, W_output]
            predicted_data = np.column_stack([
                X_phase[:, 0],  # Temperature
                X_phase[:, 1],  # Input component
                y_phase_pred    # Output component (predicted)
            ])
            
            # Create evaluator
            evaluator = PhysicalConsistencyEvaluator(
                input_boundary_model=self.input_boundary_model,
                output_boundary_model=self.output_boundary_model
            )
            
            # Evaluate physics consistency
            physics_score, results = evaluator.evaluate_full(
                dnn_model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                predicted_data=predicted_data,
                device=self.config.device
            )
            
            self.logger.info(f"Physics consistency score: {physics_score:.4f}")
            self.logger.info(f"  Boundary score: {results.get('boundary_score', 'N/A'):.4f}")
            self.logger.info(f"  Smoothness score: {results.get('smoothness_score', 'N/A'):.4f}")
            
            return {
                'physics_score': physics_score,
                'physics_boundary': results.get('boundary_score', None),
                'physics_smoothness': results.get('smoothness_score', None)
            }
        
        except Exception as e:
            self.logger.error(f"Physics evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _save_fold_results(
        self,
        fold_idx: int,
        fold_dir: Path,
        metrics: Dict,
        physics_results: Dict
    ) -> None:
        """
        Save fold results to files.
        
        Args:
            fold_idx: Fold index.
            fold_dir: Directory to save results.
            metrics: Evaluation metrics.
            physics_results: Physics evaluation results.
        """
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        
        # Save predictions
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)
    
    def _save_fold_metrics(
        self,
        metrics: Dict,
        physics_results: Dict,
        fold_dir: Path
    ) -> None:
        """Save fold metrics to Excel."""
        data = []
        
        # Extract metrics
        inner_metrics = metrics.get('metrics', metrics)
        
        # Training metrics
        data.append(['Train RMSE', inner_metrics.get('train_rmse', float('nan'))])
        data.append(['Train MAE', inner_metrics.get('train_mae', float('nan'))])
        data.append(['Train R²', inner_metrics.get('train_r2', float('nan'))])
        
        # Validation metrics
        data.append(['Val RMSE', inner_metrics.get('val_rmse', float('nan'))])
        data.append(['Val MAE', inner_metrics.get('val_mae', float('nan'))])
        data.append(['Val R²', inner_metrics.get('val_r2', float('nan'))])
        
        # Test metrics
        data.append(['Test RMSE', inner_metrics.get('test_rmse', float('nan'))])
        data.append(['Test MAE', inner_metrics.get('test_mae', float('nan'))])
        data.append(['Test R²', inner_metrics.get('test_r2', float('nan'))])
        
        # Physics metrics
        if physics_results:
            data.append(['Physics Score', physics_results.get('physics_score', float('nan'))])
            data.append(['Boundary Consistency', physics_results.get('physics_boundary', float('nan'))])
            data.append(['Thermodynamic Smoothness', physics_results.get('physics_smoothness', float('nan'))])
        
        df = pd.DataFrame(data, columns=['Metric', 'Value'])
        
        metrics_path = fold_dir / f'{self.config.excel_prefix}metrics.xlsx'
        df.to_excel(metrics_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Metrics saved: {metrics_path.name}")
    
    def _save_predictions(
        self,
        metrics: Dict,
        fold_dir: Path
    ) -> None:
        """Save predictions to Excel."""
        predictions = metrics.get('predictions', {})
        true_values = metrics.get('true_values', {})
        
        for split_name in ['train', 'val', 'test']:
            if split_name in predictions and split_name in true_values:
                y_true = true_values[split_name].flatten()
                y_pred = predictions[split_name].flatten()
                
                df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'residual': y_true - y_pred
                })
                
                pred_path = fold_dir / f'{self.config.excel_prefix}{split_name}_predictions.xlsx'
                df.to_excel(pred_path, index=False, engine='openpyxl')


# ==============================================================================
# K-Fold Experiment Manager
# ==============================================================================

class KFoldExperimentManager:
    """
    Manager for K-fold cross-validation experiments with data cleaning.
    
    Orchestrates TKMeansLOF cleaning, DNN training, and result aggregation
    across multiple folds.
    
    Workflow:
        1. Load training data
        2. Apply TKMeansLOF cleaning to all training data
        3. Split cleaned data into K folds
        4. Train and evaluate on each fold
        5. Aggregate and report results
    
    Attributes:
        config: Experiment configuration.
        n_folds: Number of cross-validation folds.
        logger: Logger instance.
        all_results: List storing results from all folds.
        output_dir: Main output directory.
        cleaning_info: Data cleaning statistics.
    """
    
    def __init__(
        self,
        config: CleaningConfig,
        n_folds: int = 5
    ) -> None:
        """
        Initialize experiment manager.
        
        Args:
            config: Configuration object.
            n_folds: Number of folds for cross-validation.
        
        Raises:
            ValueError: If n_folds < 2.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        
        self.config = config
        self.n_folds = n_folds
        self.logger = get_logger(
            self.__class__.__name__,
            level=config.log_level
        )
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f'./cleaning_results_{timestamp}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_results: List[Dict[str, Any]] = []
        self.cleaning_info: Dict[str, Any] = {}
        
        # Load boundary models
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        self.logger.info(f"Initialized K-fold manager (n_folds={n_folds})")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _load_boundary_models(self) -> None:
        """Load binary boundary models for physics evaluation."""
        try:
            input_model_path = self.config.input_boundary_model_path
            output_model_path = self.config.output_boundary_model_path
            
            if input_model_path.exists() and output_model_path.exists():
                self.input_boundary_model = BinaryPredictor.load(str(input_model_path))
                self.output_boundary_model = BinaryPredictor.load(str(output_model_path))
                self.logger.info("Boundary models loaded successfully")
            else:
                self.logger.warning("Boundary models not found, physics evaluation will be skipped")
        
        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")
    
    def run_all_folds(self) -> None:
        """
        Run complete K-fold cross-validation experiment.
        
        Steps:
            1. Load data
            2. Clean training data with TKMeansLOF
            3. Create K-folds from cleaned data
            4. Run experiment for each fold
            5. Aggregate and save summary results
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Data Cleaning Ablation Experiment - K-Fold CV")
        self.logger.info("="*70)
        self.logger.info(f"K-Folds: {self.n_folds}")
        self.logger.info(f"Random Seed: {self.config.kfold_random_state}")
        self.logger.info(f"Device: {self.config.device}")
        
        start_time = time.time()
        
        # Load and clean data
        self._load_and_clean_data()
        
        # Create folds
        self._create_folds()
        
        # Run each fold
        for fold_idx, fold_data in enumerate(self.folds):
            self._run_single_fold(fold_idx, fold_data)
        
        # Generate summary
        self._generate_summary()
        
        total_time = time.time() - start_time
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("All experiments completed successfully")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Total folds: {len(self.folds)}")
        self.logger.info(f"Results saved in: {self.output_dir}")
        self.logger.info("")
    
    def _load_and_clean_data(self) -> None:
        """Load training data and apply TKMeansLOF cleaning."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Step 1: Loading and Cleaning Data")
        self.logger.info("="*70)
        
        # Load training data (low temperature)
        ternary_path = self.config.ternary_data_path
        X_ternary, y_ternary = load_ternary_data(ternary_path)
        self.logger.info(f"Training data loaded: {len(X_ternary)} samples")
        
        # Load test data (high temperature)
        test_path = self.config.test_data_path
        X_test, y_test = load_ternary_data(test_path)
        self.logger.info(f"Test data loaded: {len(X_test)} samples")
        self.test_data = (X_test, y_test)
        
        # Apply TKMeansLOF cleaning if enabled
        if self.config.enable_cleaning:
            self.logger.info("\nApplying TKMeansLOF cleaning...")
            self.logger.info(f"Cleaning configuration:")
            for key, value in self.config.cleaning_config.items():
                self.logger.info(f"  {key}: {value}")
            
            # Prepare 3-column input [T, w_comp1, w_comp2]
            X_3col = np.column_stack([X_ternary, y_ternary])
            
            # Create cleaner
            cleaner = TKMeansLOF(
                config=self.config.cleaning_config,
                system_type='solubility',
                verbose=True
            )
            
            # Clean data
            X_clean_3col, _, outliers, clean_info = cleaner.clean(X_3col, y_ternary)
            
            # Extract back to 2-column format
            X_clean = X_clean_3col[:, :2]  # [T, w_comp1]
            y_clean = X_clean_3col[:, 2]   # w_comp2
            
            n_removed = clean_info['n_outliers']
            removal_rate = clean_info['outlier_rate']
            
            self.cleaning_info = {
                'n_original': len(X_ternary),
                'n_clean': len(X_clean),
                'n_removed': n_removed,
                'removal_rate': removal_rate,
                'outlier_indices': outliers.tolist() if hasattr(outliers, 'tolist') else list(outliers),
                'iterations': clean_info['iterations'],
                'method': 'TKMeansLOF'
            }
            
            self.logger.info(f"\nCleaning completed:")
            self.logger.info(f"  Original: {len(X_ternary)} samples")
            self.logger.info(f"  Cleaned: {len(X_clean)} samples")
            self.logger.info(f"  Removed: {n_removed} samples ({removal_rate*100:.2f}%)")
            self.logger.info(f"  Iterations: {clean_info['iterations']}")
            
            self.cleaned_data = (X_clean, y_clean)
            
            # Save cleaning results
            self._save_cleaning_info()
            
        else:
            self.logger.info("\nTKMeansLOF cleaning disabled, using original data")
            self.cleaned_data = (X_ternary, y_ternary)
            self.cleaning_info = {
                'n_original': len(X_ternary),
                'n_clean': len(X_ternary),
                'n_removed': 0,
                'removal_rate': 0.0,
                'outlier_indices': [],
                'iterations': 0,
                'method': 'None (Cleaning Disabled)'
            }
    
    def _create_folds(self) -> None:
        """Create K-fold splits from cleaned data."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Step 2: Creating K-Fold Splits")
        self.logger.info("="*70)
        
        X_clean, y_clean = self.cleaned_data
        
        self.logger.info(f"Creating K-fold splits (K={self.n_folds}, random_state={self.config.kfold_random_state})...")
        self.folds = create_k_folds(
            X_clean, y_clean,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state
        )
        
        self.logger.info(f"\nK-fold splits created:")
        for fold_idx, fold in enumerate(self.folds):
            self.logger.info(
                f"  Fold {fold_idx}: "
                f"train={len(fold['X_train'])} samples, "
                f"val={len(fold['X_val'])} samples"
            )
    
    def _save_cleaning_info(self) -> None:
        """Save data cleaning information."""
        cleaning_dir = self.output_dir / 'cleaning'
        cleaning_dir.mkdir(exist_ok=True)
        
        # Save cleaning results JSON
        cleaning_json = {
            'cleaning_enabled': self.config.enable_cleaning,
            'cleaning_method': self.cleaning_info.get('method', 'Unknown'),
            'cleaning_config': self.config.cleaning_config,
            'results': self.cleaning_info,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cleaning_dir / 'cleaning_info.json', 'w', encoding='utf-8') as f:
            json.dump(cleaning_json, f, indent=2, ensure_ascii=False)
        
        # Save removed points
        if self.cleaning_info['n_removed'] > 0:
            X_ternary, y_ternary = load_ternary_data(self.config.ternary_data_path)
            removed_idx = self.cleaning_info['outlier_indices']
            
            df_removed = pd.DataFrame({
                'index': removed_idx,
                'Temperature': X_ternary[removed_idx, 0],
                'Input_Component': X_ternary[removed_idx, 1],
                'Output_Component': y_ternary[removed_idx]
            })
            df_removed.to_excel(cleaning_dir / 'removed_points.xlsx', index=False, engine='openpyxl')
        
        # Save cleaned data
        X_clean, y_clean = self.cleaned_data
        df_clean = pd.DataFrame({
            'Temperature': X_clean[:, 0],
            'Input_Component': X_clean[:, 1],
            'Output_Component': y_clean
        })
        df_clean.to_excel(cleaning_dir / 'cleaned_data.xlsx', index=False, engine='openpyxl')
        
        self.logger.info(f"Cleaning info saved to: {cleaning_dir}")
    
    def _run_single_fold(
        self,
        fold_idx: int,
        fold_data: Dict[str, np.ndarray]
    ) -> None:
        """
        Run experiment for a single fold.
        
        Args:
            fold_idx: Fold index.
            fold_data: Fold data dictionary.
        """
        self.logger.info("")
        self.logger.info("█"*70)
        self.logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")
        self.logger.info("█"*70)
        
        # Create fold directory
        fold_dir = self.output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        # Extract test data
        X_test, y_test = self.test_data
        
        # Run fold experiment
        runner = SingleFoldRunner(
            config=self.config,
            input_boundary_model=self.input_boundary_model,
            output_boundary_model=self.output_boundary_model
        )
        
        fold_results = runner.run(
            fold_idx=fold_idx,
            X_train=fold_data['X_train'],
            y_train=fold_data['y_train'],
            X_val=fold_data['X_val'],
            y_val=fold_data['y_val'],
            X_test=X_test,
            y_test=y_test,
            fold_dir=fold_dir
        )
        
        # Store results
        self.all_results.append(fold_results)
    
    def _generate_summary(self) -> None:
        """Generate and save summary results across all folds."""
        self.logger.info("")
        self.logger.info("Generating summary results...")
        
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        # Save summary metrics
        self._save_summary_metrics(summary_dir)
        
        # Save summary predictions
        self._save_summary_predictions(summary_dir)
        
        # Generate text report
        self._generate_text_report(summary_dir)
        
        self.logger.info(f"Summary saved in: {summary_dir}")
    
    def _save_summary_metrics(self, summary_dir: Path) -> None:
        """Save aggregated metrics across folds."""
        data = []
        
        # ML metrics
        metric_keys = [
            'train_r2', 'train_rmse', 'train_mae',
            'val_r2', 'val_rmse', 'val_mae',
            'test_r2', 'test_rmse', 'test_mae'
        ]
        
        metric_names = [
            'Train R²', 'Train RMSE', 'Train MAE',
            'Val R²', 'Val RMSE', 'Val MAE',
            'Test R²', 'Test RMSE', 'Test MAE'
        ]
        
        for key, name in zip(metric_keys, metric_names):
            values = [
                r['metrics'][key]
                for r in self.all_results
            ]
            data.append([name, np.mean(values)])
        
        # Physics metrics
        physics_score_values = [
            r.get('physics_score', float('nan'))
            for r in self.all_results
        ]
        physics_score_values = [
            v for v in physics_score_values
            if v is not None and not np.isnan(v)
        ]
        
        if len(physics_score_values) > 0:
            data.append(['Physics Score', np.mean(physics_score_values)])
        
        physics_boundary_values = [
            r.get('physics_boundary', float('nan'))
            for r in self.all_results
        ]
        physics_boundary_values = [
            v for v in physics_boundary_values
            if v is not None and not np.isnan(v)
        ]
        
        if len(physics_boundary_values) > 0:
            data.append(['Boundary Consistency', np.mean(physics_boundary_values)])
        
        physics_smoothness_values = [
            r.get('physics_smoothness', float('nan'))
            for r in self.all_results
        ]
        physics_smoothness_values = [
            v for v in physics_smoothness_values
            if v is not None and not np.isnan(v)
        ]
        
        if len(physics_smoothness_values) > 0:
            data.append(['Thermodynamic Smoothness', np.mean(physics_smoothness_values)])
        
        df = pd.DataFrame(data, columns=['Metric', 'Mean Value'])
        
        summary_path = summary_dir / 'summary_metrics.xlsx'
        df.to_excel(summary_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Summary metrics saved: {summary_path.name}")
    
    def _save_summary_predictions(self, summary_dir: Path) -> None:
        """Save aggregated predictions across folds."""
        y_test = self.all_results[0]['true_values']['test'].flatten()
        
        data = {'y_true': y_test}
        
        for result in self.all_results:
            fold_idx = result['fold_idx']
            y_pred = result['predictions']['test'].flatten()
            data[f'y_pred_fold{fold_idx}'] = y_pred
        
        y_test_preds = np.array([
            r['predictions']['test'].flatten()
            for r in self.all_results
        ])
        data['y_pred_mean'] = np.mean(y_test_preds, axis=0)
        data['y_pred_std'] = np.std(y_test_preds, axis=0, ddof=1)
        data['residual_mean'] = y_test - data['y_pred_mean']
        
        df = pd.DataFrame(data)
        
        summary_path = summary_dir / 'test_predictions_summary.xlsx'
        df.to_excel(summary_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Summary predictions saved: {summary_path.name}")
    
    def _generate_text_report(self, summary_dir: Path) -> None:
        """Generate comprehensive text report."""
        report = []
        report.append("="*70)
        report.append("Data Cleaning Ablation - K-Fold CV Summary")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"K-Folds: {self.n_folds}")
        report.append(f"Random Seed: {self.config.kfold_random_state}")
        report.append(f"Device: {self.config.device}")
        
        # Cleaning information
        report.append("\n" + "-"*70)
        report.append("Data Cleaning Information")
        report.append("-"*70)
        report.append(f"Cleaning Enabled: {self.config.enable_cleaning}")
        report.append(f"Cleaning Method: {self.cleaning_info.get('method', 'Unknown')}")
        if self.cleaning_info:
            report.append(f"Original Data: {self.cleaning_info['n_original']} samples")
            report.append(f"Cleaned Data: {self.cleaning_info['n_clean']} samples")
            report.append(f"Removed: {self.cleaning_info['n_removed']} samples ({self.cleaning_info['removal_rate']*100:.2f}%)")
            report.append(f"Iterations: {self.cleaning_info['iterations']}")
        
        if self.config.enable_cleaning:
            report.append(f"\nCleaning Configuration:")
            for key, value in self.config.cleaning_config.items():
                report.append(f"  {key}: {value}")
        
        # Model configuration
        report.append("\n" + "-"*70)
        report.append("Model Configuration")
        report.append("-"*70)
        report.append(f"DNN Epochs: {self.config.dnn_epochs}")
        report.append(f"Learning Rate: {self.config.dnn_learning_rate}")
        report.append(f"Network Layers: {self.config.dnn_layer_dim}")
        report.append(f"Nodes per Layer: {self.config.dnn_node_dim}")
        
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
            values = [r['metrics'][key] for r in self.all_results]
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
        
        test_r2_values = [r['metrics']['test_r2'] for r in self.all_results]
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
    Main entry point for data cleaning ablation experiment.
    
    Configures and runs K-fold cross-validation with TKMeansLOF cleaning.
    """
    # Create configuration
    config = CleaningConfig()
    
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
    config.kfold_random_state = 42
    config.enable_cleaning = True
    config.dnn_epochs = 1000
    config.save_predictions = True
    config.save_metrics = True
    config.log_level = logging.INFO
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Data Cleaning Ablation Experiment v1.0 (Standardized)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-Folds: {config.k_folds}")
    logger.info(f"Random Seed: {config.kfold_random_state}")
    logger.info(f"Evaluation Method: TSTR (Train on Synthetic, Test on Real)")
    logger.info("")
    logger.info("Features:")
    logger.info("  - TKMeansLOF cleaning for outlier removal")
    logger.info("  - K-fold cross-validation for robust evaluation")
    logger.info("  - Physical consistency evaluation using boundary models")
    logger.info("  - Complete metrics tracking and reporting")
    logger.info("")
    logger.info("Cleaning Configuration:")
    for key, value in config.cleaning_config.items():
        logger.info(f"  {key}: {value}")
    
    # Run K-fold experiment
    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("Data cleaning experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()