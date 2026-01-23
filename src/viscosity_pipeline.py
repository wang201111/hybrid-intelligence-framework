"""
================================================================================
Viscosity Prediction Pipeline - Complete Framework Integration
================================================================================

This module provides a unified pipeline for ternary viscosity system prediction,
integrating three core modules:

1. TKMeansLOF - Data cleaning with temperature-based clustering
2. IADAFTrainer - Data augmentation with Bayesian-optimized WGAN-GP
3. EdgeSamplingCorrectedModel - Physical constraints with edge sampling correction

The pipeline supports two training modes:
- Fixed split: fit(X_train, y_train, X_val, y_val)
- K-fold cross-validation: fit_kfold(X_all, y_all)

Data Format:
- Input: X = [Temperature/°C, Pressure/(N/m²), comp1, comp2]  (4 columns)
- Output: y = Viscosity/(mPa·s)  (1 column)

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
from t_kmeas_lof import TKMeansLOF, OutlierDetectionConfig
from IADAF import IADAFTrainer, IADAFConfig
from ldpc_viscosity import EdgeSamplingCorrectedModel
from binary_predictor import BinaryPredictor

warnings.filterwarnings('ignore')


# ==============================================================================
# Configuration Class
# ==============================================================================

@dataclass
class ViscosityPipelineConfig:
    """
    Configuration parameters for ViscosityPipeline.
    
    This configuration maintains 100% backward compatibility with M4Pipeline_viscosity.
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
        
        Edge Sampling Correction Configuration:
            n_edge_samples: Number of sampling points per edge (50-100 recommended).
            edge_weight_eps: Distance weight smoothing parameter (0.001-0.1 recommended).
            use_edge_cache: Whether to cache edge corrections for performance.
        
        Binary Edge Models (for LDPC physical constraints):
            edge_CA_model_path: Path to edge CA binary model (.pth file).
            edge_CB_model_path: Path to edge CB binary model (.pth file).
            edge_AB_model_path: Path to edge AB binary model (.pth file).
        
        Data Range:
            T_min: Minimum temperature for data generation.
            T_max: Maximum temperature for data generation.
            P_min: Minimum pressure for data generation.
            P_max: Maximum pressure for data generation.
        
        Other:
            device: Computing device ('auto', 'cuda', 'cpu').
            verbose: Whether to print detailed logs.
    
    Examples:
        >>> config = ViscosityPipelineConfig(
        ...     use_kfold=True,
        ...     k_folds=5,
        ...     enable_cleaning=True,
        ...     n_samples_to_generate=2000,
        ...     edge_CA_model_path='models/binary/edge_CA.pth',
        ...     edge_CB_model_path='models/binary/edge_CB.pth',
        ...     edge_AB_model_path='models/binary/edge_AB.pth'
        ... )
        >>> pipeline = ViscosityPipeline(config)
    """
    
    # ========== Training Mode ==========
    use_kfold: bool = False
    k_folds: int = 5
    kfold_random_state: int = 42
    training_seed: int = 42
    
    # ========== TKMeansLOF Configuration ==========
    enable_cleaning: bool = True
    kmeans_lof_config: Dict = field(default_factory=lambda: {
        'lof_contamination': 0.05,
        'lof_n_neighbors': 5,
        'min_cluster_size': 10,
        'auto_select_k': True,
        'min_n_clusters': 2,
        'max_n_clusters': 24,
        'fixed_n_clusters': None,
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
        'WGAN_EPOCHS': 1000,
        'BO_WGAN_EPOCHS': 50,
        'BO_N_ITERATIONS': 100,
        'BO_INIT_POINTS': 10,
        'BO_DNN_EPOCHS': 200,
        'LATENT_DIM_RANGE': (1, 100),
        'HIDDEN_DIM_RANGE': (150, 350),
        'LAMBDA_GP_RANGE': (0.05, 5.0),
        'XGB_ERROR_THRESHOLD': 0.5,
        'XGB_MAX_DEPTH': 6,
        'XGB_N_ESTIMATORS': 100,
        'XGB_LEARNING_RATE': 0.1,
        'TRAIN_RATIO': 0.8,
        'VAL_RATIO': 0.2,
        'RANDOM_SEED': 42,
        'VERBOSE': True,
        'DEVICE': 'auto'
    })
    
    # ========== LDPC Configuration ==========
    ldpc_dnn_layers: int = 4
    ldpc_dnn_nodes: int = 128
    ldpc_batch_size: int = 1000
    ldpc_epochs: int = 1000
    ldpc_lr: float = 0.00831
    ldpc_weight_decay: float = 0.0
    ldpc_use_lr_scheduler: bool = True
    ldpc_lr_scheduler_type: str = 'cosine'
    ldpc_lr_min: float = 1e-5
    ldpc_early_stop: int = 500
    
    # ========== Edge Sampling Correction Configuration ==========
    n_edge_samples: int = 50
    edge_weight_eps: float = 0.01
    use_edge_cache: bool = True
    
    # ========== Binary Edge Models (Generic Naming) ==========
    edge_CA_model_path: Optional[Path] = None
    edge_CB_model_path: Optional[Path] = None
    edge_AB_model_path: Optional[Path] = None
    
    # ========== Data Range ==========
    T_min: float = 20.0
    T_max: float = 120.0
    P_min: float = 1e5
    P_max: float = 1e7
    
    # ========== Other ==========
    device: str = 'auto'
    verbose: bool = True
    
    def __post_init__(self):
        """Auto-detect device if set to 'auto'."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# DNN Model (4-dimensional input for viscosity)
# ==============================================================================

class DNN(nn.Module):
    """
    Deep neural network for viscosity prediction.
    
    Architecture: Input(4) → Hidden Layers → Output(1)
    
    Attributes:
        inDim: Input dimension (4: T, P, comp1, comp2).
        outDim: Output dimension (1: viscosity).
        layerDim: Number of hidden layers.
        nodeDim: Number of nodes per hidden layer.
    """
    
    def __init__(
        self,
        inDim: int = 4,
        outDim: int = 1,
        layerDim: int = 4,
        nodeDim: int = 128
    ):
        """
        Initialize DNN.
        
        Args:
            inDim: Input dimension (default: 4 for viscosity).
            outDim: Output dimension (default: 1).
            layerDim: Number of hidden layers.
            nodeDim: Number of nodes per hidden layer.
        """
        super(DNN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(inDim, nodeDim), nn.ReLU(), nn.ReLU()]
        
        # Hidden layers
        for _ in range(layerDim - 1):
            layers.extend([
                nn.Linear(nodeDim, nodeDim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # Output layer
        layers.append(nn.Linear(nodeDim, outDim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


# ==============================================================================
# Main Pipeline Class
# ==============================================================================

class ViscosityPipeline:
    """
    Complete viscosity prediction pipeline integrating three core modules.
    
    This class provides a unified interface that maintains 100% backward
    compatibility with M4Pipeline_viscosity while using the standardized modules.
    
    Workflow:
        1. TKMeansLOF: Clean experimental data (remove outliers)
        2. IADAFTrainer: Generate high-quality synthetic data
        3. EdgeSamplingCorrectedModel: Train LDPC model with edge corrections
    
    Attributes:
        config: Pipeline configuration.
        device: Computing device.
        is_fitted: Whether pipeline has been trained.
        current_mode: Training mode ('fixed' or 'kfold').
        cleaner: TKMeansLOF instance.
        augmenter: IADAFTrainer instance.
        ldpc_model: Final LDPC model with edge corrections.
        fold_models: List of models from K-fold CV.
        x_scaler: Input data scaler.
        y_scaler: Output data scaler.
        history: Training history dictionary.
    
    Examples:
        Fixed split mode:
        >>> config = ViscosityPipelineConfig()
        >>> pipeline = ViscosityPipeline(config)
        >>> pipeline.fit(X_train, y_train, X_val, y_val)
        >>> model = pipeline.get_trained_model()
        
        K-fold cross-validation mode:
        >>> config = ViscosityPipelineConfig(use_kfold=True, k_folds=5)
        >>> pipeline = ViscosityPipeline(config)
        >>> pipeline.fit_kfold(X_all, y_all)
        >>> fold_models = pipeline.get_all_fold_models()
    """
    
    def __init__(self, config: ViscosityPipelineConfig):
        """
        Initialize ViscosityPipeline.
        
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
            print("Viscosity Pipeline Initialized".center(70))
            print("=" * 70)
            print(f"Module 1: TKMeansLOF (Data Cleaning)")
            print(f"Module 2: IADAFTrainer (Data Augmentation)")
            print(f"Module 3: EdgeSamplingCorrectedModel (LDPC + Edge Correction)")
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
            X_train: Training features [N, 4] (Temperature, Pressure, comp1, comp2).
            y_train: Training targets [N] (Viscosity).
            X_val: Validation features [M, 4]. If None, splits from X_train.
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
            X_all: All features [N, 4] (Temperature, Pressure, comp1, comp2).
            y_all: All targets [N] (Viscosity).
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> config = ViscosityPipelineConfig(use_kfold=True, k_folds=5)
            >>> pipeline = ViscosityPipeline(config)
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
            X: Input features [N, 4].
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
        
        # Initialize TKMeansLOF for viscosity system
        self.cleaner = TKMeansLOF(
            config=self.config.kmeans_lof_config,
            system_type='viscosity',
            verbose=self.config.verbose
        )
        
        # Clean data
        # TKMeansLOF for viscosity expects X as [N, 4] (T, P, comp1, comp2)
        X_cleaned, y_cleaned, outlier_indices, info = self.cleaner.clean(X, y)
        
        self.X_cleaned = X_cleaned  # [N_clean, 4]
        self.y_cleaned = y_cleaned  # [N_clean]
        
        if self.config.verbose:
            print(f"  Cleaning complete: {len(X)} → {len(self.X_cleaned)} samples")
            print(f"  Outliers removed: {info['n_outliers']} ({info['outlier_rate']*100:.1f}%)")
    
    def _augment_data(self):
        """
        Augment data using IADAFTrainer.
        """
        if self.config.verbose:
            print("\n[2/3] IADAF Data Augmentation...")
        
        # Prepare data: IADAFTrainer expects [T, P, comp1, comp2, viscosity]
        # For viscosity system: X=[T, P, comp1, comp2], y=viscosity
        # Combine to 5 columns
        X_input = np.column_stack([self.X_cleaned, self.y_cleaned])
        
        # Create IADAF configuration
        cfg = IADAFConfig()
        for key, value in self.config.iadaf_config.items():
            setattr(cfg, key, value)
        cfg.DEVICE = self.config.device
        cfg.SYSTEM_TYPE = 'viscosity'
        cfg.N_SAMPLES_TO_GENERATE = self.config.n_samples_to_generate
        cfg.T_MIN = self.config.T_min
        cfg.T_MAX = self.config.T_max
        cfg.P_MIN = self.config.P_min
        cfg.P_MAX = self.config.P_max
        
        # Initialize and train IADAF
        self.augmenter = IADAFTrainer(cfg)
        self.augmenter.fit(X_input)
        
        # Generate synthetic data
        X_gen = self.augmenter.generate(
            n_samples=self.config.n_samples_to_generate,
            T_range=(self.config.T_min, self.config.T_max)
        )
        
        # Separate to X and y
        self.X_synthetic = X_gen[:, :4]  # T, P, comp1, comp2
        self.y_synthetic = X_gen[:, 4]    # Viscosity
        
        if self.config.verbose:
            print(f"  Generated {len(self.X_synthetic)} synthetic samples")
    
    def _train_ldpc(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Train LDPC model with edge sampling correction.
        
        Args:
            X_val: Validation features [M, 4].
            y_val: Validation targets [M].
        """
        if self.config.verbose:
            print("\n[3/3] LDPC Model Training...")
        
        # Standardization
        self.x_scaler = StandardScaler().fit(self.X_synthetic)
        self.y_scaler = StandardScaler().fit(self.y_synthetic.reshape(-1, 1))
        
        X_train_scaled = self.x_scaler.transform(self.X_synthetic)
        y_train_scaled = self.y_scaler.transform(
            self.y_synthetic.reshape(-1, 1)
        ).flatten()
        X_val_scaled = self.x_scaler.transform(X_val)
        y_val_scaled = self.y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Create DNN (4-dimensional input for viscosity)
        base_model = DNN(
            inDim=4,
            outDim=1,
            layerDim=self.config.ldpc_dnn_layers,
            nodeDim=self.config.ldpc_dnn_nodes
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            base_model.parameters(),
            lr=self.config.ldpc_lr,
            weight_decay=self.config.ldpc_weight_decay
        )
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler = None
        if self.config.ldpc_use_lr_scheduler:
            if self.config.ldpc_lr_scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.ldpc_epochs,
                    eta_min=self.config.ldpc_lr_min
                )
        
        # Prepare training data
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_scaled).to(self.device)
        
        # Training loop with early stopping
        best_loss = float('inf')
        best_state = None
        patience = 0
        
        for epoch in range(self.config.ldpc_epochs):
            # Training
            base_model.train()
            y_pred = base_model(X_train_tensor).squeeze()
            loss = criterion(y_pred, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Validation
            base_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)
                val_pred = base_model(X_val_tensor).squeeze()
                val_loss = criterion(val_pred, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(base_model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.config.ldpc_early_stop:
                    if self.config.verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        base_model.load_state_dict(best_state)
        
        if self.config.verbose:
            print(f"  ✓ Base DNN training complete (Best Val Loss={best_loss:.6f})")
        
        # Apply edge sampling correction if paths provided
        if (self.config.edge_CA_model_path and 
            self.config.edge_CB_model_path and 
            self.config.edge_AB_model_path):
            try:
                self.ldpc_model = self._create_edge_corrected_model(base_model)
                if self.config.verbose:
                    print("  ✓ LDPC model with edge sampling correction applied")
            except Exception as e:
                if self.config.verbose:
                    print(f"  ⚠ Edge correction failed: {e}")
                    print("  ✓ Using base DNN model")
                self.ldpc_model = base_model
        else:
            self.ldpc_model = base_model
            if self.config.verbose:
                print("  ✓ Using base DNN model (no edge models specified)")
    
    def _create_edge_corrected_model(
        self,
        base_model: nn.Module
    ) -> EdgeSamplingCorrectedModel:
        """
        Create LDPC model with edge sampling corrections.
        
        Args:
            base_model: Trained baseline DNN model.
        
        Returns:
            EdgeSamplingCorrectedModel with edge corrections.
        """
        # Load binary edge models
        edge_CA_ensemble = BinaryPredictor.load(str(self.config.edge_CA_model_path))
        edge_CB_ensemble = BinaryPredictor.load(str(self.config.edge_CB_model_path))
        edge_AB_ensemble = BinaryPredictor.load(str(self.config.edge_AB_model_path))
        
        # Create teacher dictionary
        # EdgeSamplingCorrectedModel expects keys:
        # 'MCH_HMN', 'MCH_cis_Decalin', 'cis_Decalin_HMN'
        teachers = {
            'MCH_HMN': edge_AB_ensemble,           # Edge AB
            'MCH_cis_Decalin': edge_CB_ensemble,   # Edge CB
            'cis_Decalin_HMN': edge_CA_ensemble    # Edge CA
        }
        
        # Create EdgeSamplingCorrectedModel
        corrected_model = EdgeSamplingCorrectedModel(
            base_model=base_model,
            teachers=teachers,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            device=self.config.device,
            n_edge_samples=self.config.n_edge_samples,
            eps=self.config.edge_weight_eps,
            use_cache=self.config.use_edge_cache
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
    print("\nViscosity Pipeline Usage Example")
    print("=" * 70)
    
    # Simulate data
    np.random.seed(42)
    X = np.random.rand(200, 4) * 100  # [N, 4]: T, P, comp1, comp2
    y = np.random.rand(200) * 10       # [N]: Viscosity
    
    # Create configuration
    config = ViscosityPipelineConfig(
        use_kfold=False,
        enable_cleaning=True,
        kmeans_lof_config={
            'lof_contamination': 0.05,
        },
        n_samples_to_generate=2000,
        ldpc_epochs=100,
        verbose=True,
        device='cpu'
    )
    
    # Train pipeline
    pipeline = ViscosityPipeline(config)
    pipeline.fit(X, y)
    
    # Get trained model
    ldpc_model = pipeline.get_trained_model()
    print(f"\n✓ Trained model: {type(ldpc_model).__name__}")
    
    # Save artifacts
    pipeline.save_artifacts('./pipeline_output')
    print("\n" + "=" * 70)