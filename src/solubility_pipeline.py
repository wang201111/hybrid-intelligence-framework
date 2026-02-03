"""
================================================================================
Solubility Prediction Pipeline - Complete Framework Integration
================================================================================

This module provides a unified pipeline for ternary solubility system prediction,
integrating three core modules:

1. TKMeansLOF - Data cleaning with temperature-based clustering
2. IADAFTrainer - Data augmentation with Bayesian-optimized WGAN-GP
3. ConformingCorrectedModel - Physical constraints with boundary correction

The pipeline supports two training modes:
- Fixed split: fit(X_train, y_train, X_val, y_val)
- K-fold cross-validation: fit_kfold(X_all, y_all)


================================================================================
"""

import copy
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import core modules
from t_kmeans_lof import TKMeansLOF, OutlierDetectionConfig
from iadaf import IADAFTrainer, IADAFConfig
from ldpc_solubility import ConformingCorrectedModel
from binary_predictor import BinaryPredictor

# Import training utilities
from utils_solubility import DNN, PhysicsConfig, TSTREvaluator

warnings.filterwarnings('ignore')


# ==============================================================================
# Configuration Class
# ==============================================================================

@dataclass
class SolubilityPipelineConfig:
    """
    Configuration parameters for SolubilityPipeline.
    
    This configuration maintains 100% backward compatibility with M4Pipeline.
    All parameter names and default values are preserved from the original.
    
    Attributes:
        Training Mode:
            use_kfold: Whether to use K-fold cross-validation.
            k_folds: Number of folds for K-fold CV.
            kfold_random_state: Random state for K-fold splits.
            training_seed: Random seed for training.
        
        TKMeansLOF Configuration:
            enable_cleaning: Whether to enable data cleaning.
            kmeans_lof_config: Configuration dict for TKMeansLOF module.
                Compatible with OutlierDetectionConfig parameters.
        
        IADAF Configuration:
            n_samples_to_generate: Number of synthetic samples to generate.
            iadaf_config: Configuration dict for IADAF module.
                Compatible with IADAFConfig parameters.
        
        LDPC Configuration:
            ldpc_dnn_layers: Number of DNN hidden layers.
            ldpc_dnn_nodes: Number of nodes per hidden layer.
            ldpc_batch_size: Batch size for training.
            ldpc_epochs: Number of training epochs.
            ldpc_lr: Learning rate.
            ldpc_weight_decay: Weight decay (L2 regularization).
            ldpc_use_lr_scheduler: Whether to use learning rate scheduler.
            ldpc_lr_scheduler_type: LR scheduler type ('cosine', 'step', etc.).
            ldpc_lr_min: Minimum learning rate for scheduler.
            ldpc_early_stop: Early stopping patience.
        
        Binary Boundary Models (for LDPC physical constraints):
            input_boundary_model_path: Path to input component binary model (.pth file).
            output_boundary_model_path: Path to output component binary model (.pth file).
            input_boundary_temp_range: Valid temperature range for input boundary model.
            output_boundary_temp_range: Valid temperature range for output boundary model.
            input_boundary_output_range: Valid output range for input boundary model.
            output_boundary_output_range: Valid output range for output boundary model.
        
        Other:
            T_min: Minimum temperature for data generation.
            T_max: Maximum temperature for data generation.
            device: Computing device ('auto', 'cuda', 'cpu').
            verbose: Whether to print detailed logs.
    
    Examples:
        >>> config = SolubilityPipelineConfig(
        ...     use_kfold=True,
        ...     k_folds=5,
        ...     enable_cleaning=True,
        ...     n_samples_to_generate=1500,
        ...     input_boundary_model_path='models/binary/input_component.pth',
        ...     output_boundary_model_path='models/binary/output_component.pth'
        ... )
        >>> pipeline = SolubilityPipeline(config)
    """
    
    # ========== Training Mode ==========
    use_kfold: bool = False
    k_folds: int = 5
    kfold_random_state: int = 42
    training_seed: int = 42
    
    # ========== TKMeansLOF Configuration ==========
    enable_cleaning: bool = True
    kmeans_lof_config: Dict = field(default_factory=lambda: {
        'lof_contamination': 0.20,
        'lof_n_neighbors': 5,
        'min_cluster_size': 10,
        'auto_select_k': True,
        'min_n_clusters': 5,
        'max_n_clusters': 24,
        'random_state': 42
    })
    
    # ========== IADAF Configuration ==========
    n_samples_to_generate: int = 2000
    iadaf_config: Dict = field(default_factory=lambda: {
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
        'RANDOM_SEED': 42,
        'VERBOSE': True,
        'DEVICE': 'auto',
        'LAMBDA_GP_RANGE': (0.05, 0.5),
        'LATENT_DIM_RANGE': (1, 100),
        'HIDDEN_DIM_RANGE': (150, 350),
    })
    
    # ========== LDPC Configuration ==========
    ldpc_dnn_layers: int = 4
    ldpc_dnn_nodes: int = 128
    ldpc_batch_size: int = 1000
    ldpc_epochs: int = 1000
    ldpc_lr: float = 0.008311
    ldpc_weight_decay: float = 0.0
    ldpc_use_lr_scheduler: bool = True
    ldpc_lr_scheduler_type: str = 'cosine'
    ldpc_lr_min: float = 1e-5
    ldpc_early_stop: int = 500
    
    # ========== Binary Boundary Models (Generic Naming) ==========
    input_boundary_model_path: Optional[Path] = None
    output_boundary_model_path: Optional[Path] = None
    input_boundary_temp_range: Tuple[float, float] = (-60.0, 200.0)
    output_boundary_temp_range: Tuple[float, float] = (-60.0, 200.0)
    input_boundary_output_range: Tuple[float, float] = (0.0, 80.0)
    output_boundary_output_range: Tuple[float, float] = (0.0, 80.0)
    
    # ========== Other ==========
    T_min: float = -40.0
    T_max: float = 200.0
    device: str = 'auto'
    verbose: bool = True
    
    def __post_init__(self):
        """Auto-detect device if set to 'auto'."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# Main Pipeline Class
# ==============================================================================

class SolubilityPipeline:
    """
    Complete solubility prediction pipeline integrating three core modules.
    
    This class provides a unified interface that maintains 100% backward
    compatibility with M4Pipeline while using the standardized modules.
    
    Workflow:
        1. TKMeansLOF: Clean experimental data (remove outliers)
        2. IADAFTrainer: Generate high-quality synthetic data
        3. ConformingCorrectedModel: Train LDPC model with boundary constraints
    
    Attributes:
        config: Pipeline configuration.
        device: Computing device.
        is_fitted: Whether pipeline has been trained.
        current_mode: Training mode ('fixed' or 'kfold').
        cleaner: TKMeansLOF instance.
        augmenter: IADAFTrainer instance.
        ldpc_model: Final LDPC model with boundary corrections.
        fold_models: List of models from K-fold CV.
        x_scaler: Input data scaler.
        y_scaler: Output data scaler.
        history: Training history dictionary.
    
    Examples:
        Fixed split mode:
        >>> config = SolubilityPipelineConfig()
        >>> pipeline = SolubilityPipeline(config)
        >>> pipeline.fit(X_train, y_train, X_val, y_val)
        >>> model = pipeline.get_trained_model()
        
        K-fold cross-validation mode:
        >>> config = SolubilityPipelineConfig(use_kfold=True, k_folds=5)
        >>> pipeline = SolubilityPipeline(config)
        >>> pipeline.fit_kfold(X_all, y_all)
        >>> fold_models = pipeline.get_all_fold_models()
    """
    
    def __init__(self, config: SolubilityPipelineConfig):
        """
        Initialize SolubilityPipeline.
        
        Args:
            config: Pipeline configuration object.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.is_fitted = False
        self.current_mode = None
        
        # Core modules
        self.cleaner = None
        self.augmenter = None
        
        # Data storage
        self.X_cleaned = None
        self.y_cleaned = None
        self.X_synthetic = None
        self.y_synthetic = None
        
        # LDPC model
        self.ldpc_model = None
        self.fold_models = []
        
        # Data scalers
        self.x_scaler = None
        self.y_scaler = None
        
        # Training history
        self.history = {}
        
        if config.verbose:
            print("=" * 70)
            print("Solubility Pipeline Initialized".center(70))
            print("=" * 70)
            print(f"Module 1: TKMeansLOF (Data Cleaning)")
            print(f"Module 2: IADAFTrainer (Data Augmentation)")
            print(f"Module 3: ConformingCorrectedModel (LDPC + Boundary Correction)")
            print("=" * 70)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ):
        """
        Train pipeline with fixed train/validation split.
        
        Args:
            X_train: Training features [N, 2] (Temperature, w_input).
            y_train: Training targets [N] (w_output).
            X_val: Validation features [M, 2]. If None, splits from X_train.
            y_val: Validation targets [M]. If None, splits from y_train.
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> pipeline.fit(X_train, y_train, X_val, y_val)
            >>> # Or let it auto-split:
            >>> pipeline.fit(X_train, y_train)
        """
        # Auto-split if validation data not provided
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        self.current_mode = 'fixed'
        
        # Step 1: Clean data
        self._clean_data(X_train, y_train)
        
        # Step 2: Augment data
        self._augment_data()
        
        # Step 3: Train LDPC model
        self._train_ldpc(X_val, y_val)
        
        self.is_fitted = True
        return self
    
    def fit_kfold(self, X_all: np.ndarray, y_all: np.ndarray):
        """
        Train pipeline with K-fold cross-validation.
        
        Args:
            X_all: All features [N, 2] (Temperature, w_input).
            y_all: All targets [N] (w_output).
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> config = SolubilityPipelineConfig(use_kfold=True, k_folds=5)
            >>> pipeline = SolubilityPipeline(config)
            >>> pipeline.fit_kfold(X_all, y_all)
        """
        self.current_mode = 'kfold'
        
        # Clean all data first
        self._clean_data(X_all, y_all)
        X_cleaned = self.X_cleaned
        y_cleaned = self.y_cleaned
        
        # K-fold split
        kf = KFold(
            n_splits=self.config.k_folds,
            shuffle=True,
            random_state=self.config.kfold_random_state
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cleaned)):
            if self.config.verbose:
                print(f"\nFold {fold_idx+1}/{self.config.k_folds}")
            
            # Split data for this fold
            self.X_cleaned = X_cleaned[train_idx]
            self.y_cleaned = y_cleaned[train_idx]
            X_val_fold = X_cleaned[val_idx]
            y_val_fold = y_cleaned[val_idx]
            
            # Augment and train for this fold
            self._augment_data()
            self._train_ldpc(X_val_fold, y_val_fold)
            
            # Store fold model
            self.fold_models.append({
                'fold_idx': fold_idx,
                'model': copy.deepcopy(self.ldpc_model),
                'x_scaler': copy.deepcopy(self.x_scaler),
                'y_scaler': copy.deepcopy(self.y_scaler)
            })
        
        self.is_fitted = True
        return self
    
    def _clean_data(self, X: np.ndarray, y: np.ndarray):
        """
        Clean data using TKMeansLOF.
        
        Args:
            X: Input features [N, 2].
            y: Target values [N].
        """
        if self.config.verbose:
            print("\n[1/3] TKMeansLOF Data Cleaning...")
        
        if not self.config.enable_cleaning:
            self.X_cleaned = X.copy()
            self.y_cleaned = y.copy()
            if self.config.verbose:
                print(f"  Cleaning disabled (enable_cleaning=False)")
            return
        
        # Initialize TKMeansLOF
        self.cleaner = TKMeansLOF(
            config=self.config.kmeans_lof_config,
            system_type='solubility',
            verbose=self.config.verbose
        )
        
        # Prepare data: TKMeansLOF expects [T, w_comp1, w_comp2]
        # For ternary system: X=[T, w_input], y=w_output
        # Combine to 3 columns
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        X_3col = np.column_stack([X, y_reshaped])
        
        # Clean data
        X_cleaned_3col, y_clean, outlier_indices, info = self.cleaner.clean(
            X_3col, y
        )
        
        # Separate back to X (2 columns) and y (1 column)
        self.X_cleaned = X_cleaned_3col[:, :2]  # First 2 columns: T, w_input
        self.y_cleaned = y_clean
        
        if self.config.verbose:
            print(f"  Cleaning complete: {len(X)} → {len(self.X_cleaned)} samples")
            print(f"  Outliers removed: {info['n_outliers']} ({info['outlier_rate']*100:.1f}%)")
    
    def _augment_data(self):
        """
        Augment data using IADAFTrainer.
        """
        if self.config.verbose:
            print("\n[2/3] IADAF Data Augmentation...")
        
        # Prepare data: IADAFTrainer expects [T, w_comp1, w_comp2]
        X_input = np.column_stack([self.X_cleaned, self.y_cleaned])
        
        # Create IADAF configuration
        cfg = IADAFConfig()
        for key, value in self.config.iadaf_config.items():
            setattr(cfg, key, value)
        cfg.DEVICE = self.config.device
        cfg.SYSTEM_TYPE = 'solubility'
        cfg.N_SAMPLES_TO_GENERATE = self.config.n_samples_to_generate
        cfg.T_MIN = self.config.T_min
        cfg.T_MAX = self.config.T_max
        
        # Initialize and train IADAF
        self.augmenter = IADAFTrainer(cfg)
        self.augmenter.fit(X_input)
        
        # Generate synthetic data
        X_gen = self.augmenter.generate(
            n_samples=self.config.n_samples_to_generate,
            T_range=(self.config.T_min, self.config.T_max)
        )
        
        # Separate to X and y
        self.X_synthetic = X_gen[:, :2]  # T, w_input
        self.y_synthetic = X_gen[:, 2]    # w_output
        
        if self.config.verbose:
            print(f"  Generated {len(self.X_synthetic)} synthetic samples")
    
    def _train_ldpc(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train LDPC model using TSTREvaluator.
        
        Args:
            X_val: Validation features.
            y_val: Validation targets.
        """
        if self.config.verbose:
            print("\n[3/3] LDPC Model Training (using TSTREvaluator)...")
        
        # Create temporary test set for TSTREvaluator interface
        X_test_temp = X_val.copy()
        y_test_temp = y_val.copy()
        
        # Create TSTREvaluator
        evaluator = TSTREvaluator(
            X_val=X_val,
            y_val=y_val,
            X_test=X_test_temp,
            y_test=y_test_temp,
            X_train=self.X_synthetic,
            y_train=self.y_synthetic,
            config=PhysicsConfig(
                tstr_epochs=self.config.ldpc_epochs,
                tstr_lr=self.config.ldpc_lr,
                dnn_layer_dim=self.config.ldpc_dnn_layers,
                dnn_node_dim=self.config.ldpc_dnn_nodes,
                tstr_device=self.config.device  
            )
        )
        
        # Train model
        result = evaluator.evaluate(
            X_syn=self.X_synthetic,
            y_syn=self.y_synthetic,
            epochs=self.config.ldpc_epochs,
            verbose=self.config.verbose
        )
        
        # Extract training results
        base_model = result['model']
        self.x_scaler = result['x_scaler']
        self.y_scaler = result['y_scaler']
        
        if self.config.verbose:
            val_r2 = result['metrics']['val_r2']
            print(f"  ✓ Base DNN training complete (Val R²={val_r2:.6f})")
        
        # Apply boundary correction if paths provided
        if (self.config.input_boundary_model_path and 
            self.config.output_boundary_model_path):
            try:
                self.ldpc_model = self._create_boundary_corrected_model(
                    base_model
                )
                if self.config.verbose:
                    print("  ✓ LDPC model with boundary correction applied")
            except Exception as e:
                if self.config.verbose:
                    print(f"  ⚠ Boundary correction failed: {e}")
                    print("  ✓ Using base DNN model")
                self.ldpc_model = base_model
        else:
            self.ldpc_model = base_model
            if self.config.verbose:
                print("  ✓ Using base DNN model (no boundary models specified)")
    
    def _create_boundary_corrected_model(
        self,
        base_model: nn.Module
    ) -> ConformingCorrectedModel:
        """
        Create LDPC model with boundary corrections.
        
        Args:
            base_model: Trained baseline DNN model.
        
        Returns:
            ConformingCorrectedModel with boundary constraints.
        """
        # Load binary boundary models
        input_boundary_ensemble = BinaryPredictor.load(
            str(self.config.input_boundary_model_path)
        )
        output_boundary_ensemble = BinaryPredictor.load(
            str(self.config.output_boundary_model_path)
        )
        
        # Create adapter functions with temperature and output range clipping
        class BoundaryAdapter:
            """Adapter for binary boundary model."""
            
            def __init__(
                self,
                ensemble: BinaryPredictor,
                temp_range: Tuple[float, float],
                output_range: Tuple[float, float]
            ):
                self.ensemble = ensemble
                self.temp_range = temp_range
                self.output_range = output_range
            
            def __call__(self, X: np.ndarray) -> np.ndarray:
                """
                Predict boundary values with range clipping.
                
                Args:
                    X: Input features [N, ...]. Only first column (temperature) is used.
                
                Returns:
                    Predicted boundary values [N].
                """
                T_min, T_max = self.temp_range
                X_clipped = np.clip(X[:, 0:1], T_min, T_max)
                y_pred = self.ensemble.predict(X_clipped, return_std=False)
                W_min, W_max = self.output_range
                y_pred = np.clip(y_pred, W_min, W_max)
                return y_pred
        
        # Create adapters for both boundaries
        input_boundary_adapter = BoundaryAdapter(
            input_boundary_ensemble,
            self.config.input_boundary_temp_range,
            self.config.input_boundary_output_range
        )
        output_boundary_adapter = BoundaryAdapter(
            output_boundary_ensemble,
            self.config.output_boundary_temp_range,
            self.config.output_boundary_output_range
        )
        
        # Create ConformingCorrectedModel
        corrected_model = ConformingCorrectedModel(
            base_model=base_model,
            input_boundary_model=input_boundary_adapter,
            output_boundary_model=output_boundary_adapter,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            device=self.config.device,
            correction_decay='exponential',
            decay_rate=2.0
        )
        
        return corrected_model
    
    def get_trained_model(self) -> nn.Module:
        """
        Get trained LDPC model (for fixed split mode).
        
        Returns:
            Trained LDPC model.
        
        Raises:
            RuntimeError: If pipeline not trained or in K-fold mode.
        
        Examples:
            >>> pipeline.fit(X_train, y_train)
            >>> model = pipeline.get_trained_model()
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not trained! Call fit() first.")
        if self.current_mode == 'kfold':
            raise ValueError(
                "K-fold mode: Use get_all_fold_models() instead"
            )
        return self.ldpc_model
    
    def get_all_fold_models(self) -> List[Dict]:
        """
        Get all fold models (for K-fold mode).
        
        Returns:
            List of dictionaries containing fold models and scalers.
        
        Raises:
            RuntimeError: If pipeline not trained or not in K-fold mode.
        
        Examples:
            >>> pipeline.fit_kfold(X_all, y_all)
            >>> fold_models = pipeline.get_all_fold_models()
            >>> for fold in fold_models:
            >>>     model = fold['model']
            >>>     scaler_x = fold['x_scaler']
            >>>     scaler_y = fold['y_scaler']
        """
        if not self.is_fitted or self.current_mode != 'kfold':
            raise RuntimeError("Not trained in K-fold mode! Call fit_kfold() first.")
        return self.fold_models
    
    def get_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        """
        Get data scalers.
        
        Returns:
            Tuple of (x_scaler, y_scaler).
        
        Examples:
            >>> x_scaler, y_scaler = pipeline.get_scalers()
        """
        return self.x_scaler, self.y_scaler
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using trained LDPC model (for fixed split mode).
        
        The LDPC model includes physical boundary corrections, which are
        automatically applied during prediction.
        
        Args:
            X: Input features [N, 2] (Temperature, w_input).
        
        Returns:
            Predicted values [N] (w_output) with LDPC corrections applied.
        
        Raises:
            RuntimeError: If pipeline not fitted or in K-fold mode.
        
        Examples:
            >>> pipeline.fit(X_train, y_train, X_val, y_val)
            >>> y_pred = pipeline.predict(X_test)  # LDPC corrections applied
        
        Note:
            This method calls the ConformingCorrectedModel's forward method,
            which automatically applies exponential decay boundary corrections
            based on the distance to binary boundaries.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted! Call fit() first.")
        
        if self.current_mode == 'kfold':
            raise ValueError(
                "K-fold mode: Use predict_fold() or predict_ensemble() instead"
            )
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get LDPC model (ConformingCorrectedModel with boundary corrections)
        model = self.ldpc_model
        
        # Predict with LDPC corrections
        model.eval()
        with torch.no_grad():
            X_scaled = self.x_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            # ✅ LDPC boundary corrections applied in model.forward()
            y_pred_scaled = model(X_tensor).cpu().numpy()
            y_pred = self.y_scaler.inverse_transform(y_pred_scaled).flatten()
        
        return y_pred
    
    def predict_fold(
        self,
        X: np.ndarray,
        fold_idx: int
    ) -> np.ndarray:
        """
        Predict using a specific fold model (for K-fold mode).
        
        Args:
            X: Input features [N, 2] (Temperature, w_input).
            fold_idx: Index of fold to use (0 to k_folds-1).
        
        Returns:
            Predicted values [N] (w_output) with LDPC corrections.
        
        Raises:
            RuntimeError: If pipeline not trained in K-fold mode.
            IndexError: If fold_idx out of range.
        
        Examples:
            >>> pipeline.fit_kfold(X_all, y_all)
            >>> y_pred_fold0 = pipeline.predict_fold(X_test, fold_idx=0)
        """
        if not self.is_fitted or self.current_mode != 'kfold':
            raise RuntimeError("Not trained in K-fold mode! Call fit_kfold() first.")
        
        if fold_idx < 0 or fold_idx >= len(self.fold_models):
            raise IndexError(
                f"fold_idx {fold_idx} out of range [0, {len(self.fold_models)})"
            )
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get fold model and scalers
        fold_dict = self.fold_models[fold_idx]
        model = fold_dict['model']
        x_scaler = fold_dict['x_scaler']
        y_scaler = fold_dict['y_scaler']
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_scaled = x_scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_pred_scaled = model(X_tensor).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        
        return y_pred
    
    def predict_ensemble(
        self,
        X: np.ndarray,
        return_std: bool = False
    ):
        """
        Predict using ensemble of all fold models (for K-fold mode).
        
        Averages predictions from all folds for more robust results.
        
        Args:
            X: Input features [N, 2] (Temperature, w_input).
            return_std: Whether to return prediction uncertainty.
        
        Returns:
            If return_std=False:
                Predicted values [N] (mean across folds).
            If return_std=True:
                Tuple of (y_mean [N], y_std [N]).
        
        Raises:
            RuntimeError: If pipeline not trained in K-fold mode.
        
        Examples:
            >>> pipeline.fit_kfold(X_all, y_all)
            >>> y_pred = pipeline.predict_ensemble(X_test)
            >>> y_mean, y_std = pipeline.predict_ensemble(X_test, return_std=True)
        """
        if not self.is_fitted or self.current_mode != 'kfold':
            raise RuntimeError("Not trained in K-fold mode! Call fit_kfold() first.")
        
        # Collect predictions from all folds
        all_preds = []
        for fold_idx in range(len(self.fold_models)):
            y_pred_fold = self.predict_fold(X, fold_idx)
            all_preds.append(y_pred_fold)
        
        all_preds = np.array(all_preds)  # [n_folds, N]
        
        # Compute statistics
        y_mean = np.mean(all_preds, axis=0)  # [N]
        
        if return_std:
            y_std = np.std(all_preds, axis=0, ddof=1)  # [N]
            return y_mean, y_std
        else:
            return y_mean
    def save_artifacts(self, output_dir: str):
        """
        Save all pipeline artifacts to directory.
        
        Saves:
            - cleaned_data.npz: Cleaned training data
            - synthetic_data.npz: Generated synthetic data
            - ldpc_model.pth or folds/*.pth: Trained models
            - scalers.pkl: Data scalers
        
        Args:
            output_dir: Output directory path.
        
        Examples:
            >>> pipeline.save_artifacts('./output')
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.savez(
            out / 'cleaned_data.npz',
            X=self.X_cleaned,
            y=self.y_cleaned
        )
        np.savez(
            out / 'synthetic_data.npz',
            X=self.X_synthetic,
            y=self.y_synthetic
        )
        
        # Save models
        if self.current_mode == 'fixed':
            torch.save(
                self.ldpc_model.state_dict(),
                out / 'ldpc_model.pth'
            )
        else:
            fold_dir = out / 'folds'
            fold_dir.mkdir(exist_ok=True)
            for fold_dict in self.fold_models:
                fold_idx = fold_dict['fold_idx']
                torch.save(
                    fold_dict['model'].state_dict(),
                    fold_dir / f"fold_{fold_idx}.pth"
                )
        
        # Save scalers
        with open(out / 'scalers.pkl', 'wb') as fp:
            pickle.dump({'x': self.x_scaler, 'y': self.y_scaler}, fp)
        
        print(f"✓ Artifacts saved to: {out}")


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == '__main__':
    print("\nSolubility Pipeline Usage Example")
    print("=" * 70)
    
    # Simulate data
    np.random.seed(42)
    X = np.random.rand(200, 2) * 100  # [N, 2]: Temperature, w_input
    y = np.random.rand(200) * 50       # [N]: w_output
    
    # Create configuration
    config = SolubilityPipelineConfig(
        use_kfold=False,
        enable_cleaning=True,
        kmeans_lof_config={
            'lof_contamination': 0.10,
        },
        n_samples_to_generate=500,
        ldpc_epochs=100,
        verbose=True,
        device='cpu'
    )
    
    # Train pipeline
    pipeline = SolubilityPipeline(config)
    pipeline.fit(X, y)
    
    # Get trained model
    ldpc_model = pipeline.get_trained_model()
    print(f"\n✓ Trained model: {type(ldpc_model).__name__}")
    
    # Save artifacts
    pipeline.save_artifacts('./pipeline_output')
    print("\n" + "=" * 70)