"""
================================================================================
IADAF Ablation Study - K-Fold Cross-Validation Framework
================================================================================

Evaluates DNN performance using IADAF (Integrated Adaptive Data Augmentation
Framework) with k-fold cross-validation.

Pipeline:
    1. Bayesian Optimization → Find optimal WGAN hyperparameters
    2. WGAN-GP Training → Learn data distribution
    3. Synthetic Data Generation → Generate samples
    4. XGBoost Filtering → Quality control
    5. TSTR Evaluation → Train DNN on synthetic, test on real

Features:
    - K-fold cross-validation for robust evaluation
    - Bayesian optimization for hyperparameter tuning
    - WGAN-GP for high-quality data generation
    - XGBoost filtering for quality control
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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src directory to path for imports
# File location: experiments/solubility/ablation/augmentation_experiment.py
# Need 4 parents to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Local imports
from iadaf import IADAFTrainer, IADAFConfig
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
class AugmentationConfig:
    """
    Configuration for data augmentation ablation experiment.
    
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
        
        Training Parameters:
            dnn_epochs: Number of DNN training epochs.
            wgan_epochs: Number of WGAN-GP training epochs.
            wgan_bo_epochs: WGAN epochs during Bayesian optimization.
        
        WGAN-GP Architecture:
            wgan_latent_dim: Latent space dimension (45).
            wgan_hidden_dim: Hidden layer dimension (256).
            wgan_lambda_gp: Gradient penalty coefficient (0.1).
            wgan_n_critic: Critic update frequency (5).
            wgan_learning_rate_g: Generator learning rate (1e-4).
            wgan_learning_rate_c: Critic learning rate (1e-4).
            wgan_dropout: Dropout rate (0.1).
            wgan_batch_size: Batch size (64).
        
        Bayesian Optimization:
            bo_n_iterations: BO iteration count (100).
            bo_init_points: BO initial random samples (10).
            bo_dnn_epochs: DNN training epochs during BO (300).
            latent_dim_range: Search range for latent_dim (1, 100).
            hidden_dim_range: Search range for hidden_dim (150, 350).
            lambda_gp_range: Search range for lambda_gp (0.05, 0.5).
        
        XGBoost Filtering:
            xgb_error_threshold: Error threshold for filtering (15.0).
            xgb_max_depth: Tree depth (6).
            xgb_n_estimators: Number of trees (100).
            xgb_learning_rate: Learning rate (0.1).
        
        Data Generation:
            n_samples_to_generate: Number of synthetic samples (1000).
            t_min: Minimum temperature (-40.0).
            t_max: Maximum temperature (200.0).
        
        Data Split:
            train_ratio: Training set ratio (0.8).
            val_ratio: Validation set ratio (0.2).
        
        Cross-Validation:
            k_folds: Number of folds for cross-validation (5).
            kfold_random_state: Random seed for fold splitting (42).
        
        Output Settings:
            output_dir: Output directory for results.
            save_predictions: Whether to save predictions to Excel.
            save_metrics: Whether to save metrics to Excel.
            save_bo_results: Whether to save BO results to Excel.
            excel_prefix: Prefix for Excel output files.
        
        Device:
            device: Computing device ('auto', 'cuda', 'cpu').
            random_seed: Random seed for reproducibility (42).
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
    wgan_epochs: int = 1000
    wgan_bo_epochs: int = 50
    
    # WGAN-GP architecture
    wgan_latent_dim: int = 45
    wgan_hidden_dim: int = 256
    wgan_lambda_gp: float = 0.1
    wgan_n_critic: int = 5
    wgan_learning_rate_g: float = 1e-4
    wgan_learning_rate_c: float = 1e-4
    wgan_dropout: float = 0.1
    wgan_batch_size: int = 64
    
    # Bayesian optimization
    bo_n_iterations: int = 100
    bo_init_points: int = 10
    bo_dnn_epochs: int = 300
    latent_dim_range: Tuple[int, int] = (1, 100)
    hidden_dim_range: Tuple[int, int] = (150, 350)
    lambda_gp_range: Tuple[float, float] = (0.05, 0.5)
    
    # XGBoost filtering
    xgb_error_threshold: float = 15.0
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 100
    xgb_learning_rate: float = 0.1
    
    # Data generation
    n_samples_to_generate: int = 1000
    t_min: float = -40.0
    t_max: float = 200.0
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Cross-validation parameters
    k_folds: int = 5
    kfold_random_state: int = 42
    
    # Output settings
    output_dir: Path = Path('./results/solubility/ablation/augmentation')
    save_predictions: bool = True
    save_metrics: bool = True
    save_bo_results: bool = True
    excel_prefix: str = 'augmentation_'
    
    # Device and logging
    device: str = 'auto'
    random_seed: int = 42
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
    Runs complete IADAF experiment for a single fold.
    
    Attributes:
        config: Experiment configuration.
        logger: Logger instance.
        device: Computing device.
        input_boundary_model: Input boundary model for physics evaluation.
        output_boundary_model: Output boundary model for physics evaluation.
    """
    
    def __init__(
        self,
        config: AugmentationConfig,
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
        Run complete IADAF experiment for one fold.
        
        Steps:
            1. Train IADAF (Bayesian optimization + WGAN-GP)
            2. Generate synthetic data
            3. Filter with XGBoost
            4. Train DNN on synthetic data (TSTR)
            5. Evaluate on real data
            6. Compute physics metrics
        
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
                - 'bo_history': Bayesian optimization history
                - 'best_params': Optimal hyperparameters
        """
        self.logger.info("="*70)
        self.logger.info(f"Starting Fold {fold_idx}")
        self.logger.info("="*70)
        
        # Step 1: Train IADAF
        self.logger.info("\n[Step 1] Training IADAF model")
        iadaf_trainer, bo_history, best_params = self._train_iadaf(
            X_train, y_train, X_val, y_val
        )
        
        # Step 2: Generate synthetic data
        self.logger.info("\n[Step 2] Generating synthetic data")
        X_synthetic = self._generate_synthetic_data(iadaf_trainer)
        self.logger.info(f"Generated {len(X_synthetic)} synthetic samples")
        
        # Step 3: Train DNN on synthetic data (TSTR)
        self.logger.info("\n[Step 3] TSTR - Training DNN on synthetic data")
        dnn_model, x_scaler, y_scaler, dnn_history = self._train_dnn_tstr(
            X_synthetic, X_val, y_val, X_test, y_test
        )
        
        # Step 4: Evaluate DNN on real data
        self.logger.info("\n[Step 4] Evaluating DNN on real data")
        metrics = self._evaluate_dnn(
            dnn_model, x_scaler, y_scaler,
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 5: Compute physics metrics
        self.logger.info("\n[Step 5] Computing physics metrics")
        physics_results = self._compute_physics_metrics(
            dnn_model, x_scaler, y_scaler
        )
        
        # Save results
        self._save_fold_results(
            fold_idx, fold_dir, metrics, physics_results, 
            bo_history, best_params
        )
        
        # Combine results
        return {
            'fold_idx': fold_idx,
            'metrics': metrics,
            'physics_score': physics_results.get('physics_score', None),
            'physics_boundary': physics_results.get('physics_boundary', None),
            'physics_smoothness': physics_results.get('physics_smoothness', None),
            'bo_history': bo_history,
            'best_params': best_params
        }
    
    def _train_iadaf(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[IADAFTrainer, List[Dict], Dict]:
        """
        Train IADAF model with Bayesian optimization.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        
        Returns:
            Tuple of (trainer, bo_history, best_params).
        """
        # Prepare data for IADAF (concatenate X and y)
        X_train_full = np.column_stack([X_train, y_train])
        X_val_full = np.column_stack([X_val, y_val])
        
        # Create IADAF configuration
        iadaf_config = IADAFConfig(
            SYSTEM_TYPE='solubility',
            LATENT_DIM=self.config.wgan_latent_dim,
            HIDDEN_DIM=self.config.wgan_hidden_dim,
            LAMBDA_GP=self.config.wgan_lambda_gp,
            N_CRITIC=self.config.wgan_n_critic,
            DROPOUT=self.config.wgan_dropout,
            LEARNING_RATE_G=self.config.wgan_learning_rate_g,
            LEARNING_RATE_C=self.config.wgan_learning_rate_c,
            BATCH_SIZE=self.config.wgan_batch_size,
            WGAN_EPOCHS=self.config.wgan_epochs,
            BO_WGAN_EPOCHS=self.config.wgan_bo_epochs,
            BO_N_ITERATIONS=self.config.bo_n_iterations,
            BO_INIT_POINTS=self.config.bo_init_points,
            BO_DNN_EPOCHS=self.config.bo_dnn_epochs,
            LATENT_DIM_RANGE=self.config.latent_dim_range,
            HIDDEN_DIM_RANGE=self.config.hidden_dim_range,
            LAMBDA_GP_RANGE=self.config.lambda_gp_range,
            XGB_ERROR_THRESHOLD=self.config.xgb_error_threshold,
            XGB_MAX_DEPTH=self.config.xgb_max_depth,
            XGB_N_ESTIMATORS=self.config.xgb_n_estimators,
            XGB_LEARNING_RATE=self.config.xgb_learning_rate,
            N_SAMPLES_TO_GENERATE=self.config.n_samples_to_generate,
            T_MIN=self.config.t_min,
            T_MAX=self.config.t_max,
            TRAIN_RATIO=self.config.train_ratio,
            VAL_RATIO=self.config.val_ratio,
            RANDOM_SEED=self.config.random_seed,
            DEVICE=self.config.device,
            VERBOSE=False
        )
        
        # Train IADAF
        trainer = IADAFTrainer(iadaf_config)
        trainer.fit_with_validation(X_train_full, X_val_full)
        
        # Extract BO history and best parameters
        bo_history = trainer.history.get('bo_history', [])
        best_params = trainer.best_params
        
        self.logger.info(f"IADAF training completed")
        if best_params:
            self.logger.info(f"Best hyperparameters:")
            self.logger.info(f"  latent_dim: {best_params.get('latent_dim', 'N/A')}")
            self.logger.info(f"  hidden_dim: {best_params.get('hidden_dim', 'N/A')}")
            self.logger.info(f"  lambda_gp: {best_params.get('lambda_gp', 'N/A')}")
        
        return trainer, bo_history, best_params
    
    def _generate_synthetic_data(
        self,
        trainer: IADAFTrainer
    ) -> np.ndarray:
        """
        Generate synthetic data using trained IADAF.
        
        Args:
            trainer: Trained IADAF trainer.
        
        Returns:
            Synthetic data array [n_samples, 3].
        """
        X_synthetic = trainer.generate(n_samples=self.config.n_samples_to_generate)
        return X_synthetic
    
    def _train_dnn_tstr(
        self,
        X_synthetic: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[nn.Module, StandardScaler, StandardScaler, Dict]:
        """
        Train DNN using TSTR approach (Train on Synthetic, Test on Real).
        
        Args:
            X_synthetic: Synthetic data [n_synthetic, 3].
            X_val: Real validation features.
            y_val: Real validation targets.
            X_test: Real test features.
            y_test: Real test targets.
        
        Returns:
            Tuple of (model, x_scaler, y_scaler, history).
        """
        # Split synthetic data
        X_syn_train = X_synthetic[:, :2]
        y_syn_train = X_synthetic[:, 2]
        
        # Scale data
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_syn_scaled = x_scaler.fit_transform(X_syn_train)
        y_syn_scaled = y_scaler.fit_transform(y_syn_train.reshape(-1, 1)).ravel()
        
        X_val_scaled = x_scaler.transform(X_val)
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        
        X_test_scaled = x_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        
        # Convert to tensors
        X_syn_t = torch.FloatTensor(X_syn_scaled).to(self.device)
        y_syn_t = torch.FloatTensor(y_syn_scaled).unsqueeze(1).to(self.device)
        
        X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_t = torch.FloatTensor(y_val_scaled).unsqueeze(1).to(self.device)
        
        X_test_t = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_t = torch.FloatTensor(y_test_scaled).unsqueeze(1).to(self.device)
        
        # Create model
        model = DNN(input_dim=2, layer_dim=4, node_dim=128).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.00831)
        criterion = nn.MSELoss()
        
        # Training
        history = {
            'train_r2': [],
            'val_r2': [],
            'test_r2': []
        }
        
        best_val_r2 = -np.inf
        best_model_state = None
        
        for epoch in range(self.config.dnn_epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            y_pred_train = model(X_syn_t)
            loss = criterion(y_pred_train, y_syn_t)
            
            loss.backward()
            optimizer.step()
            
            # Evaluation every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    # Training metrics
                    train_pred_np = y_pred_train.cpu().numpy()
                    train_r2 = 1 - (
                        np.sum((y_syn_t.cpu().numpy() - train_pred_np)**2) /
                        np.sum((y_syn_t.cpu().numpy() - y_syn_t.cpu().numpy().mean())**2)
                    )
                    
                    # Validation metrics
                    val_pred = model(X_val_t).cpu().numpy()
                    val_r2 = 1 - (
                        np.sum((y_val_t.cpu().numpy() - val_pred)**2) /
                        np.sum((y_val_t.cpu().numpy() - y_val_t.cpu().numpy().mean())**2)
                    )
                    
                    # Test metrics
                    test_pred = model(X_test_t).cpu().numpy()
                    test_r2 = 1 - (
                        np.sum((y_test_t.cpu().numpy() - test_pred)**2) /
                        np.sum((y_test_t.cpu().numpy() - y_test_t.cpu().numpy().mean())**2)
                    )
                    
                    history['train_r2'].append(train_r2)
                    history['val_r2'].append(val_r2)
                    history['test_r2'].append(test_r2)
                    
                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_model_state = model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, x_scaler, y_scaler, history
    
    def _evaluate_dnn(
        self,
        model: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate DNN on real datasets.
        
        Args:
            model: Trained DNN model.
            x_scaler: Feature scaler.
            y_scaler: Target scaler.
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            X_test: Test features.
            y_test: Test targets.
        
        Returns:
            Dictionary containing metrics and predictions.
        """
        # Use TSTREvaluator
        evaluator = TSTREvaluator(
            X_synthetic=None,  # Not needed for evaluation
            y_synthetic=None,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            device=self.device
        )
        
        # Evaluate
        results = evaluator.evaluate_model(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            X_train=X_train,
            y_train=y_train
        )
        
        return results
    
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
            evaluator = PhysicalConsistencyEvaluator(
                input_boundary_model=self.input_boundary_model,
                output_boundary_model=self.output_boundary_model
            )
            
            physics_score, results = evaluator.evaluate_full(
                dnn_model=model,
                dnn_x_scaler=x_scaler,
                dnn_y_scaler=y_scaler
            )
            
            return {
                'physics_score': physics_score,
                'physics_boundary': results.get('boundary_consistency', None),
                'physics_smoothness': results.get('thermodynamic_smoothness', None)
            }
        
        except Exception as e:
            self.logger.error(f"Physics evaluation failed: {e}")
            return {}
    
    def _save_fold_results(
        self,
        fold_idx: int,
        fold_dir: Path,
        metrics: Dict,
        physics_results: Dict,
        bo_history: List[Dict],
        best_params: Dict
    ) -> None:
        """
        Save fold results to files.
        
        Args:
            fold_idx: Fold index.
            fold_dir: Directory to save results.
            metrics: Evaluation metrics.
            physics_results: Physics evaluation results.
            bo_history: Bayesian optimization history.
            best_params: Best hyperparameters.
        """
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        if self.config.save_metrics:
            self._save_fold_metrics(metrics, physics_results, fold_dir)
        
        # Save BO results
        if self.config.save_bo_results and bo_history:
            self._save_bo_results(bo_history, best_params, fold_dir)
        
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
    
    def _save_bo_results(
        self,
        bo_history: List[Dict],
        best_params: Dict,
        fold_dir: Path
    ) -> None:
        """Save Bayesian optimization results to Excel."""
        if not bo_history:
            return
        
        data = []
        for i, entry in enumerate(bo_history):
            data.append({
                'Iteration': i,
                'latent_dim': entry.get('latent_dim', ''),
                'hidden_dim': entry.get('hidden_dim', ''),
                'lambda_gp': entry.get('lambda_gp', ''),
                'target': entry.get('target', '')
            })
        
        df = pd.DataFrame(data)
        
        # Mark best parameters
        if best_params and len(df) > 0:
            best_score = max([e.get('target', -np.inf) for e in bo_history])
            df['is_best'] = df['target'] == best_score
        
        bo_path = fold_dir / f'{self.config.excel_prefix}bo_results.xlsx'
        df.to_excel(bo_path, index=False, engine='openpyxl')
        
        self.logger.info(f"BO results saved: {bo_path.name}")
    
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
    Manager for K-fold cross-validation experiments with IADAF.
    
    Orchestrates IADAF training, evaluation, and result aggregation across
    multiple folds.
    
    Attributes:
        config: Experiment configuration.
        n_folds: Number of cross-validation folds.
        logger: Logger instance.
        all_results: List storing results from all folds.
        output_dir: Main output directory.
    """
    
    def __init__(
        self,
        config: AugmentationConfig,
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
        self.output_dir = Path(f'./augmentation_results_{timestamp}')
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
        """Load binary boundary models for physics evaluation."""
        try:
            input_model_path = self.config.models_dir / self.config.input_boundary_model_file
            output_model_path = self.config.models_dir / self.config.output_boundary_model_file
            
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
            1. Load data and create K-folds
            2. Run experiment for each fold
            3. Aggregate and save summary results
        """
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Data Augmentation Ablation Experiment - K-Fold CV")
        self.logger.info("="*70)
        self.logger.info(f"K-Folds: {self.n_folds}")
        self.logger.info(f"Random Seed: {self.config.kfold_random_state}")
        self.logger.info(f"Device: {self.config.device}")
        
        start_time = time.time()
        
        # Load data
        self._load_and_split_data()
        
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
    
    def _load_and_split_data(self) -> None:
        """Load data and create K-fold splits."""
        self.logger.info("")
        self.logger.info("Loading data and creating K-fold splits...")
        
        # Load training data (low temperature)
        ternary_path = self.config.data_dir / self.config.ternary_data_file
        X_ternary, y_ternary = load_ternary_data(ternary_path)
        self.logger.info(f"Training data loaded: {len(X_ternary)} samples")
        
        # Load test data (high temperature)
        test_path = self.config.data_dir / self.config.test_data_file
        X_test, y_test = load_ternary_data(test_path)
        self.logger.info(f"Test data loaded: {len(X_test)} samples")
        
        # Create K-folds
        self.folds = create_k_folds(
            X_ternary, y_ternary,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state
        )
        self.test_data = (X_test, y_test)
        
        self.logger.info(f"K-fold splits created:")
        for fold_idx, fold in enumerate(self.folds):
            self.logger.info(
                f"  Fold {fold_idx}: "
                f"train={len(fold['X_train'])} samples, "
                f"val={len(fold['X_val'])} samples"
            )
    
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
        
        df = pd.DataFrame(data, columns=['Metric', 'Mean Value'])
        
        summary_path = summary_dir / 'summary_metrics.xlsx'
        df.to_excel(summary_path, index=False, engine='openpyxl')
        
        self.logger.info(f"Summary metrics saved: {summary_path.name}")
    
    def _save_summary_predictions(self, summary_dir: Path) -> None:
        """Save aggregated predictions across folds."""
        y_test = self.all_results[0]['metrics']['true_values']['test'].flatten()
        
        data = {'y_true': y_test}
        
        for result in self.all_results:
            fold_idx = result['fold_idx']
            y_pred = result['metrics']['predictions']['test'].flatten()
            data[f'y_pred_fold{fold_idx}'] = y_pred
        
        y_test_preds = np.array([
            r['metrics']['predictions']['test'].flatten()
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
        report.append("Data Augmentation Ablation - K-Fold CV Summary")
        report.append("="*70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"K-Folds: {self.n_folds}")
        report.append(f"Random Seed: {self.config.kfold_random_state}")
        report.append(f"Device: {self.config.device}")
        
        # Configuration
        report.append("\n" + "-"*70)
        report.append("Configuration")
        report.append("-"*70)
        report.append(f"WGAN Epochs: {self.config.wgan_epochs}")
        report.append(f"DNN Epochs: {self.config.dnn_epochs}")
        report.append(f"BO Iterations: {self.config.bo_n_iterations}")
        report.append(f"Latent Dim: {self.config.wgan_latent_dim}")
        report.append(f"Hidden Dim: {self.config.wgan_hidden_dim}")
        report.append(f"Lambda GP: {self.config.wgan_lambda_gp}")
        
        # Summary statistics
        report.append("\n" + "-"*70)
        report.append("Summary Statistics")
        report.append("-"*70)
        
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
    Main entry point for data augmentation ablation experiment.
    
    Configures and runs K-fold cross-validation with IADAF.
    """
    # Create configuration
    config = AugmentationConfig()
    
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
    config.dnn_epochs = 1000
    config.wgan_epochs = 1000
    config.save_predictions = True
    config.save_metrics = True
    config.save_bo_results = True
    config.log_level = logging.INFO
    
    # Log experiment information
    logger.info("")
    logger.info("="*70)
    logger.info("Data Augmentation Ablation Experiment v1.0 (Standardized)")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-Folds: {config.k_folds}")
    logger.info(f"Random Seed: {config.kfold_random_state}")
    logger.info(f"Evaluation Method: TSTR (Train on Synthetic, Test on Real)")
    logger.info("")
    logger.info("Features:")
    logger.info("  - K-fold cross-validation for robust evaluation")
    logger.info("  - Bayesian optimization for hyperparameter tuning")
    logger.info("  - WGAN-GP for data generation")
    logger.info("  - XGBoost filtering for quality control")
    logger.info("  - Physical consistency evaluation using boundary models")
    logger.info("  - Complete metrics tracking and reporting")
    
    # Run K-fold experiment
    manager = KFoldExperimentManager(config, n_folds=config.k_folds)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("Data augmentation experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()