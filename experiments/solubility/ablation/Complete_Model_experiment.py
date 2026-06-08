"""
================================================================================
Complete Model (M4) Ablation Experiment - - K-Fold Cross-Validation Framework
================================================================================

Evaluates the complete M4 pipeline with all three core modules:
    1. TKMeansLOF - Data cleaning with temperature-based clustering
    2. IADAF - Data augmentation with Bayesian-optimized WGAN-GP
    3. LDPC - Physical constraints with conforming boundary correction

Pipeline:
    1. Data Loading → Load ternary system training and test data
    2. Global Cleaning → TKMeansLOF outlier detection on ALL training data
    3. K-Fold Split → Create folds from the cleaned training data
    4. For each fold:
       a. Data Augmentation → IADAF synthetic sample generation
       b. Model Training → Train baseline DNN using TSTREvaluator
       c. LDPC Correction → Wrap trained DNN with ConformingCorrectedModel
       d. Evaluation → Evaluate corrected model on validation and test sets
    5. Aggregate Results → Generate summary across all folds

Features:
    - Complete M4 pipeline with all three modules
    - K-fold cross-validation for robust evaluation
    - Physical consistency evaluation
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
# File location: experiments/solubility/ablation/complete_experiment.py
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Local imports
from solubility_pipeline import SolubilityPipeline, SolubilityPipelineConfig
from binary_predictor import BinaryPredictor
from t_kmeans_lof import TKMeansLOF
from utils_solubility import (
    DNN,
    PhysicalConsistencyEvaluator,
    PhysicsConfig,
    TSTREvaluator,
)

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
class CompleteModelConfig:
    """
    Configuration for complete M4 pipeline ablation experiment.
    
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
        
        TKMeansLOF Configuration (Data Cleaning):
            enable_cleaning: Whether to enable data cleaning.
            contamination: Expected outlier proportion (0.0-0.5).
            n_neighbors: Number of neighbors for LOF.
            min_cluster_size: Minimum cluster size.
            auto_select_k: Auto-select number of clusters.
            min_n_clusters: Minimum number of clusters.
            max_n_clusters: Maximum number of clusters.
        
        IADAF Configuration (Data Augmentation):
            n_samples_to_generate: Number of synthetic samples.
            latent_dim: Latent dimension for WGAN.
            hidden_dim: Hidden layer dimension for WGAN.
            lambda_gp: Gradient penalty coefficient.
            n_critic: Number of critic updates per generator update.
            wgan_epochs: Number of WGAN training epochs.
            bo_n_iterations: Bayesian optimization iterations.
        
        LDPC Configuration (Physical Constraints):
            correction_decay: Correction decay type ('exponential' or 'linear').
            decay_rate: Exponential decay rate.
        
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
    
    # TKMeansLOF parameters (Data Cleaning - Module 1)
    enable_cleaning: bool = True
    contamination: float = 0.20
    n_neighbors: int = 5
    min_cluster_size: int = 10
    auto_select_k: bool = True
    min_n_clusters: int = 2
    max_n_clusters: int = 24
    
    # IADAF parameters (Data Augmentation - Module 2)
    n_samples_to_generate: int = 2000
    latent_dim: int = 10
    hidden_dim: int = 256
    lambda_gp: float = 10.0
    n_critic: int = 5
    wgan_epochs: int = 500
    bo_n_iterations: int = 100
    
    # LDPC parameters (Physical Constraints - Module 3)
    correction_decay: str = 'exponential'
    decay_rate: float = 2.0
    
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
    output_dir: Path = PROJECT_ROOT / 'results' / 'solubility' / 'ablation' / 'Complete_Model_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'complete_'
    
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
    Runs complete M4 pipeline experiment for a single fold.
    
    The M4 pipeline consists of three sequential modules:
        1. TKMeansLOF - Data cleaning
        2. IADAF - Data augmentation
        3. LDPC - Physical constraints correction
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        device: Computing device.
        input_boundary_model: Input boundary model for LDPC correction.
        output_boundary_model: Output boundary model for LDPC correction.
    """
    
    def __init__(
        self,
        config: CompleteModelConfig,
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
        Run complete M4 pipeline experiment for one fold.
        
        Steps:
            1. Create SolubilityPipeline with M4 configuration
            2. Fit pipeline on training data:
               a. Clean data with TKMeansLOF
               b. Augment data with IADAF
               c. Train baseline DNN
               d. Wrap with LDPC correction
            3. Predict on validation and test sets
            4. Compute physics metrics
        
        Args:
            fold_idx: Index of current fold.
            X_train: Training features [N_train, 2].
            y_train: Training targets [N_train].
            X_val: Validation features [N_val, 2].
            y_val: Validation targets [N_val].
            X_test: Test features [N_test, 2].
            y_test: Test targets [N_test].
            fold_dir: Directory to save fold results.
        
        Returns:
            Dictionary containing:
                - 'fold_idx': Fold index
                - 'metrics': Complete evaluation metrics
                - 'physics_score': Physics evaluation score
                - 'cleaning_stats': Data cleaning statistics
                - 'augmentation_stats': Data augmentation statistics
        """
        self.logger.info("="*70)
        self.logger.info(f"Starting Fold {fold_idx} - Complete M4 Pipeline")
        self.logger.info("="*70)
        
        # Step 1: Create Solubility Pipeline
        self.logger.info("\n[Step 1] Creating M4 Pipeline")
        pipeline_result = self._create_and_fit_pipeline(
            X_train, y_train, X_val, y_val
        )
        
        pipeline = pipeline_result['pipeline']
        cleaning_stats = pipeline_result.get('cleaning_stats', {})
        augmentation_stats = pipeline_result.get('augmentation_stats', {})
        
        # Step 2: Evaluate on all datasets
        self.logger.info("\n[Step 2] Evaluating M4 pipeline")
        metrics = self._evaluate_pipeline(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 3: Compute physics metrics
        self.logger.info("\n[Step 3] Computing physics metrics")
        physics_results = self._compute_physics_metrics(pipeline)
        
        # Save results
        self._save_fold_results(
            fold_idx, fold_dir, metrics, physics_results,
            cleaning_stats, augmentation_stats
        )
        
        # Combine results
        return {
            'fold_idx': fold_idx,
            'metrics': metrics,
            'predictions': metrics.get('predictions', {}),
            'true_values': metrics.get('true_values', {}),
            'physics_score': physics_results.get('physics_score', None),
            'physics_boundary': physics_results.get('physics_boundary', None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
            'cleaning_stats': cleaning_stats,
            'augmentation_stats': augmentation_stats
        }
    
    def _create_and_fit_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create and fit the M4 pipeline.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        
        Returns:
            Dictionary with pipeline and statistics.
        """
        # Create pipeline configuration
        pipeline_config = SolubilityPipelineConfig(
            # Training mode
            use_kfold=False,  # Fixed split mode
            training_seed=self.config.random_seed,
            
            # TKMeansLOF (Data Cleaning) — DISABLED here because cleaning
            # was already applied globally before fold splitting.
            enable_cleaning=False,
            kmeans_lof_config={
                'contamination': self.config.contamination,
                'n_neighbors': self.config.n_neighbors,
                'min_cluster_size': self.config.min_cluster_size,
                'auto_select_k': self.config.auto_select_k,
                'min_n_clusters': self.config.min_n_clusters,
                'max_n_clusters': self.config.max_n_clusters,
                'random_state': self.config.random_seed
            },
            
            # IADAF (Data Augmentation)
            n_samples_to_generate=self.config.n_samples_to_generate,
            iadaf_config={
                'LATENT_DIM': self.config.latent_dim,
                'HIDDEN_DIM': self.config.hidden_dim,
                'LAMBDA_GP': self.config.lambda_gp,
                'N_CRITIC': self.config.n_critic,
                'WGAN_EPOCHS': self.config.wgan_epochs,
                'BO_N_ITERATIONS': self.config.bo_n_iterations,
            },
            
            # LDPC (Physical Constraints)
            ldpc_epochs=self.config.dnn_epochs,
            ldpc_lr=self.config.dnn_learning_rate,
            ldpc_dnn_layers=self.config.dnn_layer_dim,
            ldpc_dnn_nodes=self.config.dnn_node_dim,
            input_boundary_model_path=str(self.config.input_boundary_model_path),
            output_boundary_model_path=str(self.config.output_boundary_model_path),
            
            # Device
            device=self.config.device,
            verbose=True
        )
        
        # Create pipeline
        self.logger.info("Creating SolubilityPipeline...")
        pipeline = SolubilityPipeline(pipeline_config)
        
        # Fit pipeline
        self.logger.info("Fitting M4 pipeline...")
        self.logger.info("  Module 1: TKMeansLOF data cleaning")
        self.logger.info("  Module 2: IADAF data augmentation")
        self.logger.info("  Module 3: LDPC baseline DNN training")
        
        fit_result = pipeline.fit(X_train, y_train, X_val, y_val)
        

        cleaning_stats = {}
        if self.config.enable_cleaning:
            # Get cleaning statistics from pipeline
            n_original = len(X_train)
            n_cleaned = n_original  
            
            # Pipeline stores cleaned data in X_cleaned attribute
            if hasattr(pipeline, 'X_cleaned') and pipeline.X_cleaned is not None:
                n_cleaned = len(pipeline.X_cleaned)
            
            n_removed = n_original - n_cleaned
            removal_rate = n_removed / n_original if n_original > 0 else 0.0
            
            cleaning_stats = {
                'n_original': n_original,
                'n_cleaned': n_cleaned,
                'n_removed': n_removed,
                'removal_rate': removal_rate
            }
            
            if n_removed > 0:
                self.logger.info(f"Cleaning: {n_removed} outliers removed ({removal_rate*100:.1f}%)")
            else:
                self.logger.info(f"Cleaning: No outliers removed")
        
        augmentation_stats = {}
        if self.config.n_samples_to_generate > 0:
            # Get augmentation statistics from pipeline
            n_synthetic = self.config.n_samples_to_generate
            n_final = n_synthetic  # Default
            
            # Pipeline stores synthetic data in X_synthetic attribute
            if hasattr(pipeline, 'X_synthetic') and pipeline.X_synthetic is not None:
                n_synthetic = len(pipeline.X_synthetic)
                # Final training set size (cleaned + synthetic used for training)
                if hasattr(pipeline, 'X_cleaned') and pipeline.X_cleaned is not None:
                    n_final = len(pipeline.X_cleaned) + n_synthetic
                else:
                    n_final = len(X_train) + n_synthetic
            
            augmentation_stats = {
                'n_synthetic': n_synthetic,
                'n_final': n_final
            }
            self.logger.info(f"Augmentation: {n_synthetic} synthetic samples generated")
        
        self.logger.info("M4 pipeline fitted successfully")
        
        return {
            'pipeline': pipeline,
            'cleaning_stats': cleaning_stats,
            'augmentation_stats': augmentation_stats,
            'fit_result': fit_result
        }
    
    def _evaluate_pipeline(
        self,
        pipeline: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline on all datasets.
        
        Args:
            pipeline: Fitted SolubilityPipeline.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary containing metrics and predictions.
        """

        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        # Compute metrics
        metrics_dict = {
            'train_r2': float(r2_score(y_train, y_train_pred)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            'train_mae': float(mean_absolute_error(y_train, y_train_pred)),
            'val_r2': float(r2_score(y_val, y_val_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            'val_mae': float(mean_absolute_error(y_val, y_val_pred)),
            'test_r2': float(r2_score(y_test, y_test_pred)),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            'test_mae': float(mean_absolute_error(y_test, y_test_pred))
        }
        
        self.logger.info(f"Pipeline evaluation:")
        self.logger.info(f"  Train: RMSE={metrics_dict['train_rmse']:.4f}, R²={metrics_dict['train_r2']:.4f}")
        self.logger.info(f"  Val:   RMSE={metrics_dict['val_rmse']:.4f}, R²={metrics_dict['val_r2']:.4f}")
        self.logger.info(f"  Test:  RMSE={metrics_dict['test_rmse']:.4f}, R²={metrics_dict['test_r2']:.4f}")
        
        return {
            'metrics': metrics_dict,
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
    
    def _compute_physics_metrics(
        self,
        pipeline: Any
    ) -> Dict[str, float]:
        """
        Compute physics consistency metrics.
        
        Args:
            pipeline: Fitted SolubilityPipeline.
        
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
            

            y_phase_pred = pipeline.predict(X_phase)
            
            # Construct predicted_data [T, W_input, W_output]
            predicted_data = np.column_stack([
                X_phase[:, 0],  # Temperature
                X_phase[:, 1],  # Input component
                y_phase_pred    # Output component (predicted with LDPC)
            ])
            
            # Get model and scalers for PhysicalConsistencyEvaluator
            model = pipeline.get_trained_model()
            x_scaler, y_scaler = pipeline.get_scalers()
            
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
        physics_results: Dict,
        cleaning_stats: Dict,
        augmentation_stats: Dict
    ) -> None:
        """
        Save fold results to files.
        
        Args:
            fold_idx: Fold index.
            fold_dir: Directory to save results.
            metrics: Evaluation metrics.
            physics_results: Physics evaluation results.
            cleaning_stats: Data cleaning statistics.
            augmentation_stats: Data augmentation statistics.
        """
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if self.config.save_metrics:
            self._save_fold_metrics(
                metrics, physics_results, cleaning_stats,
                augmentation_stats, fold_dir
            )
        
        # Save predictions
        if self.config.save_predictions:
            self._save_predictions(metrics, fold_dir)
    
    def _save_fold_metrics(
        self,
        metrics: Dict,
        physics_results: Dict,
        cleaning_stats: Dict,
        augmentation_stats: Dict,
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
        
        # Cleaning statistics
        if cleaning_stats:
            data.append(['Original Samples', cleaning_stats.get('n_original', float('nan'))])
            data.append(['Cleaned Samples', cleaning_stats.get('n_cleaned', float('nan'))])
            data.append(['Removed Samples', cleaning_stats.get('n_removed', float('nan'))])
            data.append(['Removal Rate', cleaning_stats.get('removal_rate', float('nan'))])
        
        # Augmentation statistics
        if augmentation_stats:
            data.append(['Synthetic Samples', augmentation_stats.get('n_synthetic', float('nan'))])
            data.append(['Final Samples', augmentation_stats.get('n_final', float('nan'))])
        
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
    Manager for K-fold cross-validation experiments with complete M4 pipeline.
    
    Orchestrates complete M4 pipeline (cleaning + augmentation + LDPC),
    baseline DNN training, and result aggregation across multiple folds.
    
    Workflow:
        1. Load training and test data
        2. Create K-fold splits
        3. For each fold:
           - Run complete M4 pipeline
           - Evaluate performance
        4. Aggregate and report results
    
    Attributes:
        config: Experiment configuration.
        n_folds: Number of cross-validation folds.
        logger: Logger instance.
        all_results: List storing results from all folds.
        output_dir: Main output directory.
    """
    
    def __init__(
        self,
        config: CompleteModelConfig,
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
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_results: List[Dict[str, Any]] = []
        
        # Load boundary models
        self.input_boundary_model = None
        self.output_boundary_model = None
        self._load_boundary_models()
        
        self.logger.info(f"Initialized K-fold manager (n_folds={n_folds})")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _load_boundary_models(self) -> None:
        """Load binary boundary models for LDPC correction and physics evaluation."""
        try:
            input_model_path = self.config.input_boundary_model_path
            output_model_path = self.config.output_boundary_model_path
            
            if input_model_path.exists() and output_model_path.exists():
                self.input_boundary_model = BinaryPredictor.load(str(input_model_path))
                self.output_boundary_model = BinaryPredictor.load(str(output_model_path))
                self.logger.info("Boundary models loaded successfully")
            else:
                self.logger.warning("Boundary models not found, LDPC correction will be disabled")
        
        except Exception as e:
            self.logger.error(f"Failed to load boundary models: {e}")
    
    def run_all_folds(self) -> None:
        """
        Run complete K-fold cross-validation experiment.
        
        Steps:
            1. Load data and clean globally with TKMeansLOF (before any splitting)
            2. Create K-folds from the cleaned data
            3. Run M4 pipeline for each fold (cleaning disabled inside pipeline)
            4. Aggregate and save summary results
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Complete Model (M4) Ablation Experiment - K-Fold CV")
        self.logger.info("="*70)
        self.logger.info(f"K-Folds: {self.n_folds}")
        self.logger.info(f"Random Seed: {self.config.kfold_random_state}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info("")
        self.logger.info("M4 Pipeline Modules:")
        self.logger.info(f"  1. TKMeansLOF (Data Cleaning): {self.config.enable_cleaning}")
        self.logger.info(f"  2. IADAF (Data Augmentation): {self.config.n_samples_to_generate} samples")
        self.logger.info(f"  3. LDPC (Physical Constraints): decay={self.config.correction_decay}, rate={self.config.decay_rate}")
        
        start_time = time.time()
        
        # Step 1: Load and clean ALL training data globally (before fold splitting)
        self._load_and_clean_data()
        
        # Step 2: Create K-folds from the cleaned data
        self._create_folds()
        
        # Step 3: Run each fold (pipeline cleaning disabled — data already clean)
        for fold_idx, fold_data in enumerate(self.folds):
            self._run_single_fold(fold_idx, fold_data)
        
        # Step 4: Generate summary
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
        """
        Step 1: Load all training data and apply TKMeansLOF cleaning globally.
        
        Cleaning is performed on the ENTIRE training set before any fold splitting.
        This ensures consistent outlier removal across all folds, matching the
        original pipeline logic: clean first, then split.
        
        The test set is loaded separately and is never cleaned or split.
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Step 1: Loading and cleaning all training data (TKMeansLOF - Module 1)")
        self.logger.info("="*70)
        
        # Load training data (low temperature)
        X_ternary, y_ternary = load_ternary_data(self.config.ternary_data_path)
        self.logger.info(f"Training data loaded: {len(X_ternary)} samples")
        
        # Load test data (high temperature) — fixed, not cleaned
        X_test, y_test = load_ternary_data(self.config.test_data_path)
        self.logger.info(f"Test data loaded: {len(X_test)} samples (fixed, not cleaned)")
        self.test_data = (X_test, y_test)
        
        # Global TKMeansLOF cleaning on all training data
        if self.config.enable_cleaning:
            self.logger.info("")
            self.logger.info("[TKMeansLOF Global Cleaning] Cleaning all training data...")
            self.logger.info(f"  contamination: {self.config.contamination}")
            self.logger.info(f"  n_neighbors: {self.config.n_neighbors}")
            self.logger.info(f"  auto_select_k: {self.config.auto_select_k}")
            self.logger.info(f"  cluster range: [{self.config.min_n_clusters}, {self.config.max_n_clusters}]")
            
            # Build 3-column input [T, W_input, W_output] — same as original
            X_3col = np.column_stack([X_ternary, y_ternary])
            y_dummy = np.arange(len(X_3col))  # dummy target (not used by cleaner)
            
            # Configure and run TKMeansLOF
            cleaning_config = {
                'contamination': self.config.contamination,
                'n_neighbors': self.config.n_neighbors,
                'min_cluster_size': self.config.min_cluster_size,
                'auto_select_k': self.config.auto_select_k,
                'min_n_clusters': self.config.min_n_clusters,
                'max_n_clusters': self.config.max_n_clusters,
                'random_state': self.config.random_seed
            }
            
            cleaner = TKMeansLOF(config=cleaning_config, verbose=True)
            X_clean_3col, _, removed_indices, clean_info = cleaner.clean(X_3col, y_dummy)
            
            # Split back into 2-col X and 1-col y
            X_clean = X_clean_3col[:, :2]   # [T, W_input]
            y_clean = X_clean_3col[:, 2]    # W_output
            
            n_removed = clean_info['n_outliers']
            removal_rate = clean_info['outlier_rate']
            
            self.cleaning_info = {
                'n_original': len(X_ternary),
                'n_cleaned': len(X_clean),
                'n_removed': n_removed,
                'removal_rate': removal_rate,
                'removed_indices': removed_indices.tolist() if hasattr(removed_indices, 'tolist') else list(removed_indices),
                'iterations': clean_info.get('iterations', 1),
                'method': 'TKMeansLOF Global Cleaning (M4 Module 1)'
            }
            
            self.logger.info("")
            self.logger.info(f"[TKMeansLOF] Cleaning complete:")
            self.logger.info(f"  Original: {len(X_ternary)} samples")
            self.logger.info(f"  Cleaned:  {len(X_clean)} samples")
            self.logger.info(f"  Removed:  {n_removed} samples ({removal_rate*100:.2f}%)")
            
            self.cleaned_data = (X_clean, y_clean)
        else:
            self.logger.info("[TKMeansLOF Global Cleaning] Disabled — using raw data")
            self.cleaned_data = (X_ternary, y_ternary)
            self.cleaning_info = {
                'n_original': len(X_ternary),
                'n_cleaned': len(X_ternary),
                'n_removed': 0,
                'removal_rate': 0.0,
                'removed_indices': [],
                'iterations': 0,
                'method': 'None (Cleaning Disabled)'
            }
        
        # Save cleaning info
        self._save_cleaning_info()
    
    def _create_folds(self) -> None:
        """
        Step 2: Create K-fold splits from the globally cleaned data.
        
        Folds are created AFTER cleaning so that every fold's train/val
        samples are drawn from the same cleaned dataset.
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Step 2: Creating K-fold splits (from cleaned data)")
        self.logger.info("="*70)
        
        X_clean, y_clean = self.cleaned_data
        
        self.folds = create_k_folds(
            X_clean, y_clean,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state
        )
        
        self.logger.info(f"\nK-fold splits created (from {len(X_clean)} cleaned samples):")
        for fold_idx, fold in enumerate(self.folds):
            self.logger.info(
                f"  Fold {fold_idx}: "
                f"train={len(fold['X_train'])} samples, "
                f"val={len(fold['X_val'])} samples"
            )
    
    def _save_cleaning_info(self) -> None:
        """Save global cleaning information to disk."""
        cleaning_dir = self.output_dir / 'cleaning'
        cleaning_dir.mkdir(parents=True, exist_ok=True)
        
        cleaning_json = {
            'cleaning_enabled': self.config.enable_cleaning,
            'cleaning_method': self.cleaning_info.get('method', 'Unknown'),
            'cleaning_config': {
                'contamination': self.config.contamination,
                'n_neighbors': self.config.n_neighbors,
                'min_cluster_size': self.config.min_cluster_size,
                'auto_select_k': self.config.auto_select_k,
                'min_n_clusters': self.config.min_n_clusters,
                'max_n_clusters': self.config.max_n_clusters,
            },
            'results': self.cleaning_info,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cleaning_dir / 'cleaning_info.json', 'w', encoding='utf-8') as f:
            json.dump(cleaning_json, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Cleaning info saved: {cleaning_dir / 'cleaning_info.json'}")
    
    def _run_single_fold(
        self,
        fold_idx: int,
        fold_data: Dict[str, np.ndarray]
    ) -> None:
        """
        Run M4 pipeline experiment for a single fold.
        
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
                r['metrics']['metrics'][key]
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
        
        # Cleaning statistics
        cleaning_removed = [
            r.get('cleaning_stats', {}).get('n_removed', 0)
            for r in self.all_results
        ]
        if any(cleaning_removed):
            data.append(['Avg Removed Samples', np.mean(cleaning_removed)])
        
        # Augmentation statistics
        augmentation_synthetic = [
            r.get('augmentation_stats', {}).get('n_synthetic', 0)
            for r in self.all_results
        ]
        if any(augmentation_synthetic):
            data.append(['Synthetic Samples', augmentation_synthetic[0]])  # Same for all folds
        
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
        report.append("Complete Model (M4) Ablation - K-Fold CV Summary")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"K-Folds: {self.n_folds}")
        report.append(f"Random Seed: {self.config.kfold_random_state}")
        report.append(f"Device: {self.config.device}")
        
        # M4 Pipeline Configuration
        report.append("\n" + "-"*70)
        report.append("M4 Pipeline Configuration")
        report.append("-"*70)
        report.append("Module 1 - TKMeansLOF (Data Cleaning):")
        report.append(f"  Enabled: {self.config.enable_cleaning}")
        report.append(f"  Contamination: {self.config.contamination}")
        report.append(f"  Neighbors: {self.config.n_neighbors}")
        report.append("\nModule 2 - IADAF (Data Augmentation):")
        report.append(f"  Synthetic Samples: {self.config.n_samples_to_generate}")
        report.append(f"  WGAN Epochs: {self.config.wgan_epochs}")
        report.append(f"  BO Iterations: {self.config.bo_n_iterations}")
        report.append("\nModule 3 - LDPC (Physical Constraints):")
        report.append(f"  Correction Decay: {self.config.correction_decay}")
        report.append(f"  Decay Rate: {self.config.decay_rate}")
        
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
        
        # Pipeline statistics
        report.append("\nPipeline Statistics:")
        
        # Cleaning stats
        cleaning_removed = [
            r.get('cleaning_stats', {}).get('n_removed', 0)
            for r in self.all_results
        ]
        if any(cleaning_removed):
            mean_removed = np.mean(cleaning_removed)
            report.append(f"Avg Removed Samples : {mean_removed:.1f}")
        
        # Augmentation stats
        augmentation_synthetic = [
            r.get('augmentation_stats', {}).get('n_synthetic', 0)
            for r in self.all_results
        ]
        if any(augmentation_synthetic):
            report.append(f"Synthetic Samples   : {augmentation_synthetic[0]}")
        
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
    Main entry point for complete M4 pipeline ablation experiment.
    
    Configures and runs K-fold cross-validation with complete M4 pipeline
    (TKMeansLOF + IADAF + LDPC).
    """
    # Create configuration
    config = CompleteModelConfig()
    
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
    config.n_samples_to_generate = 2000
    config.dnn_epochs = 1000
    config.correction_decay = 'exponential'
    config.decay_rate = 2.0
    config.save_predictions = True
    config.save_metrics = True
    config.log_level = logging.INFO
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Complete Model (M4) Ablation Experiment v1.0 (Standardized)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-Folds: {config.k_folds}")
    logger.info(f"Random Seed: {config.kfold_random_state}")
    logger.info("")
    logger.info("M4 Pipeline Modules:")
    logger.info("  1. TKMeansLOF - Data cleaning with temperature-based clustering")
    logger.info("  2. IADAF - Data augmentation with Bayesian-optimized WGAN-GP")
    logger.info("  3. LDPC - Physical constraints with conforming boundary correction")
    logger.info("")
    logger.info("Features:")
    logger.info("  - Complete end-to-end M4 pipeline")
    logger.info("  - K-fold cross-validation for robust evaluation")
    logger.info("  - Physical consistency evaluation")
    logger.info("  - Complete metrics tracking and reporting")
    
    # Run K-fold experiment
    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("Complete model experiment finished successfully")
    logger.info("")


if __name__ == "__main__":
    main()