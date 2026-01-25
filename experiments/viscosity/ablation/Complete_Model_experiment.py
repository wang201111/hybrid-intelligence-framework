"""
Complete M4 ablation experiment for viscosity system with k-fold cross-validation.

This experiment represents the complete M4 framework with all three modules:
    Module 1: TKMeansLOF - Data cleaning (outlier removal)
    Module 2: IADAFTrainer - Data augmentation (synthetic sample generation)
    Module 3: EdgeSamplingCorrectedModel - LDPC with edge sampling correction

Experimental Design:
    1. Load low-temperature data (<40°C) for k-fold cross-validation
    2. Load high-temperature data (≥40°C) for extrapolation testing
    3. For each fold:
        - Stage 1: Clean data with TKMeansLOF
        - Stage 2: Generate synthetic samples with IADAF (Bayesian-optimized WGAN-GP)
        - Stage 3: Train LDPC model with edge sampling correction
        - Evaluate on validation and test sets
        - Physics evaluation with 3 binary boundaries
    4. Aggregate results across all folds

Viscosity System:
    - Input: 4D [Temperature, Pressure, MCH%, cis-Decalin%]
    - Output: 1D [Viscosity]
    - Binary models: MCH-HMN, MCH-cis-Decalin, cis-Decalin-HMN
    - Edge sampling correction: 3 edges
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

# Project root and path configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm

# Local imports
from viscosity_pipeline import ViscosityPipeline, ViscosityPipelineConfig
from binary_predictor import BinaryPredictor
from utils_viscosity import ViscosityPhysicsEvaluator

warnings.filterwarnings('ignore')


# ==============================================================================
# Logging Configuration
# ==============================================================================

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
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


logger = setup_logger(__name__)


# ==============================================================================
# Binary Predictor Wrapper for Physics Evaluation
# ==============================================================================

class BinaryPredictorWrapper:
    """
    Wrapper for BinaryPredictor to provide predict_physical interface.
    
    Converts 4D input [T, P, MCH, Dec] to 3D input [T, P, component]
    for binary boundary models.
    """
    
    def __init__(self, predictor: BinaryPredictor, component_index: int = 2):
        self.predictor = predictor
        self.component_index = component_index
    
    def predict_physical(self, X: np.ndarray) -> np.ndarray:
        """Predict with 4D to 3D conversion."""
        X_3d = np.column_stack([
            X[:, 0],
            X[:, 1],
            X[:, self.component_index]
        ])
        return self.predictor.predict(X_3d)
    
    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """Compatibility interface."""
        return self.predict_physical(X)


# ==============================================================================
# Utility Functions
# ==============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
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
# Data Loading
# ==============================================================================

def load_viscosity_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ternary viscosity data from Excel file.
    
    Expected format:
        Column 0: Temperature (°C)
        Column 1: Pressure (Pa)
        Column 2: MCH composition (%)
        Column 3: cis-Decalin composition (%)
        Column 4: Viscosity (mPa·s)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_excel(filepath, engine='openpyxl')
    
    if len(df.columns) < 5:
        raise ValueError(
            f"Data file must have at least 5 columns, found {len(df.columns)}"
        )
    
    X = df.iloc[:, :4].values.astype(np.float32)
    y = df.iloc[:, 4].values.astype(np.float32)
    
    logger.info(f"Loaded {filepath.name}:")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  T range: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}] °C")
    logger.info(f"  P range: [{X[:, 1].min():.2e}, {X[:, 1].max():.2e}] Pa")
    logger.info(f"  MCH range: [{X[:, 2].min():.2f}, {X[:, 2].max():.2f}] %")
    logger.info(f"  cis-Decalin range: [{X[:, 3].min():.2f}, {X[:, 3].max():.2f}] %")
    logger.info(f"  Viscosity range: [{y.min():.4f}, {y.max():.4f}] mPa·s")
    
    return X, y


def create_k_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Dict[str, np.ndarray]]:
    """Create k-fold splits for cross-validation."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in kfold.split(X):
        fold = {
            'X_train': X[train_idx].copy(),
            'y_train': y[train_idx].copy(),
            'X_val': X[val_idx].copy(),
            'y_val': y[val_idx].copy()
        }
        folds.append(fold)
    
    return folds


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class M4ViscosityConfig:
    """Configuration for M4 viscosity experiment."""
    data_dir: Path = PROJECT_ROOT / 'data' / 'viscosity' / 'split_by_temperature'
    low_temp_file: str = 'low_temp.xlsx'
    high_temp_file: str = 'high_temp.xlsx'
    
    models_dir: Path = PROJECT_ROOT / 'models' / 'viscosity' / 'binary'
    mch_hmn_model_file: str = 'MCH_HMN.pth'
    mch_dec_model_file: str = 'MCH_cis_Decalin.pth'
    dec_hmn_model_file: str = 'cis_Decalin_HMN.pth'
    
    k_folds: int = 5
    kfold_random_state: int = 42
    
    enable_cleaning: bool = True
    cleaning_config: Dict = field(default_factory=lambda: {
        'lof_contamination': 0.05,
        'lof_n_neighbors': 5,
        'min_cluster_size': 10,
        'auto_select_k': True,
        'min_n_clusters': 2,
        'max_n_clusters': 24,
        'fixed_n_clusters': None,
        'random_state': 42
    })
    
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
        'BO_N_ITERATIONS': 50,
        'BO_INIT_POINTS': 10,
        'BO_DNN_EPOCHS': 200,
        'LATENT_DIM_RANGE': (1, 100),
        'HIDDEN_DIM_RANGE': (150, 350),
        'LAMBDA_GP_RANGE': (0.05, 0.5),
        'XGB_ERROR_THRESHOLD': 15,
        'XGB_MAX_DEPTH': 6,
        'XGB_N_ESTIMATORS': 100,
        'XGB_LEARNING_RATE': 0.1,
        'TRAIN_RATIO': 0.8,
        'VAL_RATIO': 0.2,
        'RANDOM_SEED': 42,
        'VERBOSE': False,
        'DEVICE': 'auto'
    })
    
    ldpc_dnn_layers: int = 4
    ldpc_dnn_nodes: int = 128
    ldpc_epochs: int = 1000
    ldpc_lr: float = 0.008311
    ldpc_batch_size: int = 1000
    ldpc_weight_decay: float = 0.0
    ldpc_use_lr_scheduler: bool = True
    ldpc_lr_scheduler_type: str = 'cosine'
    ldpc_lr_min: float = 1e-5
    ldpc_early_stop: int = 500
    
    n_edge_samples: int = 50
    edge_weight_eps: float = 0.01
    use_edge_cache: bool = True
    
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    t_min: float = 20.0
    t_max: float = 80.0
    p_min: float = 1e6
    p_max: float = 1e8
    
    teacher_temp_range: Tuple[float, float] = (20.0, 80.0)
    teacher_pressure_range: Tuple[float, float] = (1e5, 1e8)
    
    output_dir: Path = PROJECT_ROOT / 'results' / 'viscosity' / 'ablation' / 'Complete_Model_results'
    save_predictions: bool = True
    save_metrics: bool = True
    excel_prefix: str = 'Complete_Model_'
    
    random_seed: int = 42
    device: str = 'auto'
    verbose: bool = True
    log_level: int = logging.INFO
    
    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# Metrics Saving
# ==============================================================================

def save_fold_metrics_to_excel(
    metrics_dict: Dict[str, Any],
    physics_results: Optional[Dict[str, Any]],
    save_path: Path,
    logger_instance: Optional[logging.Logger] = None
) -> None:
    """Save fold metrics to Excel file."""
    if logger_instance is None:
        logger_instance = logger
    
    data = []
    
    data.append(['Train RMSE', metrics_dict.get('train_rmse', float('nan'))])
    data.append(['Train MAE', metrics_dict.get('train_mae', float('nan'))])
    data.append(['Train R²', metrics_dict.get('train_r2', float('nan'))])
    
    data.append(['Val RMSE', metrics_dict.get('val_rmse', float('nan'))])
    data.append(['Val MAE', metrics_dict.get('val_mae', float('nan'))])
    data.append(['Val R²', metrics_dict.get('val_r2', float('nan'))])
    
    data.append(['Test RMSE', metrics_dict.get('test_rmse', float('nan'))])
    data.append(['Test MAE', metrics_dict.get('test_mae', float('nan'))])
    data.append(['Test R²', metrics_dict.get('test_r2', float('nan'))])
    
    if physics_results is not None:
        data.append(['Physics Score', physics_results.get('overall_score', float('nan'))])
        data.append(['Boundary Consistency', physics_results.get('boundary_score', float('nan'))])
        data.append(['Thermodynamic Smoothness', physics_results.get('smoothness_score', float('nan'))])
    
    df = pd.DataFrame(data, columns=['Metric', 'Value'])
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(save_path, index=False, engine='openpyxl')
    
    logger_instance.debug(f"Metrics saved: {save_path.name}")


def save_predictions_to_excel(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    save_dir: Path,
    prefix: str = ''
) -> None:
    """Save predictions to Excel files."""
    excel_dir = save_dir / 'excel'
    excel_dir.mkdir(parents=True, exist_ok=True)
    
    df_test = pd.DataFrame({
        'Temperature': X_test[:, 0],
        'Pressure': X_test[:, 1],
        'MCH': X_test[:, 2],
        'cis_Decalin': X_test[:, 3],
        'Viscosity_True': y_test.flatten(),
        'Viscosity_Pred': y_test_pred.flatten(),
        'Error': y_test.flatten() - y_test_pred.flatten()
    })
    df_test.to_excel(excel_dir / f'{prefix}test_predictions.xlsx', index=False, engine='openpyxl')
    
    df_train = pd.DataFrame({
        'Temperature': X_train[:, 0],
        'Pressure': X_train[:, 1],
        'MCH': X_train[:, 2],
        'cis_Decalin': X_train[:, 3],
        'Viscosity_True': y_train.flatten(),
        'Viscosity_Pred': y_train_pred.flatten(),
        'Error': y_train.flatten() - y_train_pred.flatten()
    })
    df_train.to_excel(excel_dir / f'{prefix}train_predictions.xlsx', index=False, engine='openpyxl')
    
    df_val = pd.DataFrame({
        'Temperature': X_val[:, 0],
        'Pressure': X_val[:, 1],
        'MCH': X_val[:, 2],
        'cis_Decalin': X_val[:, 3],
        'Viscosity_True': y_val.flatten(),
        'Viscosity_Pred': y_val_pred.flatten(),
        'Error': y_val.flatten() - y_val_pred.flatten()
    })
    df_val.to_excel(excel_dir / f'{prefix}val_predictions.xlsx', index=False, engine='openpyxl')


# ==============================================================================
# Configuration Saving
# ==============================================================================

def save_config(
    metrics: Dict[str, Any],
    physics_results: Optional[Dict[str, Any]],
    config: M4ViscosityConfig,
    save_dir: Path,
    fold_idx: int
) -> None:
    """Save experiment configuration to JSON."""
    config_dict = {
        'experiment_info': {
            'experiment_name': 'M4 Complete Framework - Viscosity System',
            'fold_index': fold_idx,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': config.device
        },
        'model_architecture': {
            'type': 'M4: TKMeansLOF + IADAF + LDPC with Edge Sampling',
            'enable_cleaning': config.enable_cleaning,
            'n_samples_to_generate': config.n_samples_to_generate,
            'epochs': config.ldpc_epochs,
            'learning_rate': config.ldpc_lr,
            'layer_dim': config.ldpc_dnn_layers,
            'node_dim': config.ldpc_dnn_nodes,
            'n_edge_samples': config.n_edge_samples,
            'edge_weight_eps': config.edge_weight_eps
        },
        'k_fold_config': {
            'n_folds': config.k_folds,
            'random_state': config.kfold_random_state,
            'current_fold': fold_idx
        },
        'iadaf_config': {
            'wgan_epochs': config.iadaf_config['WGAN_EPOCHS'],
            'bo_iterations': config.iadaf_config['BO_N_ITERATIONS'],
            'latent_dim': config.iadaf_config['LATENT_DIM'],
            'hidden_dim': config.iadaf_config['HIDDEN_DIM']
        },
        'physics_config': {
            'boundary_decay_lambda': config.boundary_decay_lambda,
            'smoothness_decay_lambda': config.smoothness_decay_lambda
        },
        'metrics': metrics
    }
    
    if physics_results is not None:
        config_dict['physics_results'] = {
            'overall_score': physics_results.get('overall_score'),
            'boundary_score': physics_results.get('boundary_score'),
            'smoothness_score': physics_results.get('smoothness_score')
        }
    
    config_dict = convert_numpy_types(config_dict)
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


# ==============================================================================
# Pipeline Training and Evaluation
# ==============================================================================

class PipelineWrapper:
    """Wrapper for ViscosityPipeline to provide predict() interface."""
    
    def __init__(self, pipeline: ViscosityPipeline):
        self.pipeline = pipeline
        self.model = pipeline.get_trained_model()
        self.x_scaler, self.y_scaler = pipeline.get_scalers()
        self.device = pipeline.device
    
    def predict(self, X: np.ndarray, return_original_scale: bool = True) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features [N, 4].
            return_original_scale: If True, return predictions in original scale.
                                   If False, return normalized predictions.
        
        Returns:
            Predictions array [N,].
        """
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                y_pred_scaled = self.model.predict(X_tensor).cpu().numpy().flatten()
            else:
                y_pred_scaled = self.model(X_tensor).cpu().numpy().flatten()
        
        if return_original_scale:
            y_pred = self.y_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
        else:
            y_pred = y_pred_scaled
        
        return y_pred
    
    def predict_physical(self, X: np.ndarray) -> np.ndarray:
        """Physics evaluation interface (same as predict with return_original_scale=True)."""
        return self.predict(X, return_original_scale=True)


def train_m4_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: M4ViscosityConfig,
    save_dir: Path
) -> Dict[str, Any]:
    """Train M4 pipeline with all three modules."""
    logger.info("="*70)
    logger.info("Training M4 Complete Framework")
    logger.info("="*70)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    pipeline_config = ViscosityPipelineConfig(
        use_kfold=False,
        enable_cleaning=config.enable_cleaning,
        kmeans_lof_config=config.cleaning_config,
        n_samples_to_generate=config.n_samples_to_generate,
        iadaf_config=config.iadaf_config,
        ldpc_dnn_layers=config.ldpc_dnn_layers,
        ldpc_dnn_nodes=config.ldpc_dnn_nodes,
        ldpc_batch_size=config.ldpc_batch_size,
        ldpc_epochs=config.ldpc_epochs,
        ldpc_lr=config.ldpc_lr,
        ldpc_weight_decay=config.ldpc_weight_decay,
        ldpc_use_lr_scheduler=config.ldpc_use_lr_scheduler,
        ldpc_lr_scheduler_type=config.ldpc_lr_scheduler_type,
        ldpc_lr_min=config.ldpc_lr_min,
        ldpc_early_stop=config.ldpc_early_stop,
        n_edge_samples=config.n_edge_samples,
        edge_weight_eps=config.edge_weight_eps,
        use_edge_cache=config.use_edge_cache,
        edge_CA_model_path=config.models_dir / config.dec_hmn_model_file,
        edge_CB_model_path=config.models_dir / config.mch_hmn_model_file,
        edge_AB_model_path=config.models_dir / config.mch_dec_model_file,
        T_min=config.t_min,
        T_max=config.t_max,
        P_min=config.p_min,
        P_max=config.p_max,
        device=config.device,
        verbose=False
    )
    
    logger.info("Initializing ViscosityPipeline...")
    pipeline = ViscosityPipeline(pipeline_config)
    
    logger.info("Training M4 pipeline:")
    logger.info("  [1/3] TKMeansLOF data cleaning...")
    logger.info("  [2/3] IADAF data augmentation (WGAN-GP + BO)...")
    logger.info("  [3/3] LDPC training with edge sampling correction...")
    
    pipeline.fit(X_train, y_train, X_val, y_val)
    
    logger.info("Training completed")
    
    wrapper = PipelineWrapper(pipeline)
    
    logger.info("Making predictions on all sets...")
    y_train_pred = wrapper.predict(X_train)
    y_val_pred = wrapper.predict(X_val)
    y_test_pred = wrapper.predict(X_test)
    
    metrics = {
        'train_r2': sklearn_metrics.r2_score(y_train, y_train_pred),
        'train_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_train, y_train_pred)),
        'train_mae': sklearn_metrics.mean_absolute_error(y_train, y_train_pred),
        'val_r2': sklearn_metrics.r2_score(y_val, y_val_pred),
        'val_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_val, y_val_pred)),
        'val_mae': sklearn_metrics.mean_absolute_error(y_val, y_val_pred),
        'test_r2': sklearn_metrics.r2_score(y_test, y_test_pred),
        'test_rmse': np.sqrt(sklearn_metrics.mean_squared_error(y_test, y_test_pred)),
        'test_mae': sklearn_metrics.mean_absolute_error(y_test, y_test_pred)
    }
    
    logger.info("="*70)
    logger.info("Evaluation Results")
    logger.info("="*70)
    logger.info(f"Train R²:  {metrics['train_r2']:.6f}")
    logger.info(f"Val R²:    {metrics['val_r2']:.6f}")
    logger.info(f"Test R²:   {metrics['test_r2']:.6f}")
    logger.info(f"Train RMSE: {metrics['train_rmse']:.6f}")
    logger.info(f"Val RMSE:   {metrics['val_rmse']:.6f}")
    logger.info(f"Test RMSE:  {metrics['test_rmse']:.6f}")
    logger.info("="*70)
    
    return {
        'pipeline': pipeline,
        'wrapper': wrapper,
        'metrics': metrics,
        'predictions': {
            'train': y_train_pred,
            'val': y_val_pred,
            'test': y_test_pred
        }
    }


# ==============================================================================
# Physics Evaluation
# ==============================================================================

def run_physics_evaluation(
    wrapper: PipelineWrapper,
    config: M4ViscosityConfig,
    save_dir: Path
) -> Optional[Dict[str, Any]]:
    """Run physics consistency evaluation."""
    logger.info("")
    logger.info("="*70)
    logger.info("Physics Consistency Evaluation")
    logger.info("="*70)
    
    mch_hmn_path = config.models_dir / config.mch_hmn_model_file
    mch_dec_path = config.models_dir / config.mch_dec_model_file
    dec_hmn_path = config.models_dir / config.dec_hmn_model_file
    
    logger.info("Loading binary boundary models...")
    logger.info(f"  MCH-HMN: {mch_hmn_path}")
    logger.info(f"  MCH-Dec: {mch_dec_path}")
    logger.info(f"  Dec-HMN: {dec_hmn_path}")
    
    try:
        if not mch_hmn_path.exists():
            raise FileNotFoundError(f"MCH-HMN model not found: {mch_hmn_path}")
        if not mch_dec_path.exists():
            raise FileNotFoundError(f"MCH-Dec model not found: {mch_dec_path}")
        if not dec_hmn_path.exists():
            raise FileNotFoundError(f"Dec-HMN model not found: {dec_hmn_path}")
        
        teacher_mch_hmn_raw = BinaryPredictor.load(str(mch_hmn_path))
        teacher_mch_dec_raw = BinaryPredictor.load(str(mch_dec_path))
        teacher_dec_hmn_raw = BinaryPredictor.load(str(dec_hmn_path))
        
        logger.info("All binary models loaded successfully")
        
        teacher_mch_hmn = BinaryPredictorWrapper(teacher_mch_hmn_raw, component_index=2)
        teacher_dec_hmn = BinaryPredictorWrapper(teacher_dec_hmn_raw, component_index=3)
        teacher_mch_dec = BinaryPredictorWrapper(teacher_mch_dec_raw, component_index=2)
        
        logger.info("Binary models wrapped with predict_physical interface")
        
        evaluator = ViscosityPhysicsEvaluator(
            teacher_models=(teacher_mch_hmn, teacher_dec_hmn, teacher_mch_dec),
            temp_range=config.teacher_temp_range,
            pressure_range=config.teacher_pressure_range,
            boundary_decay_lambda=config.boundary_decay_lambda,
            smoothness_decay_lambda=config.smoothness_decay_lambda,
            log_level=config.log_level
        )
        
        overall_score, results = evaluator.evaluate_full(wrapper)
        
        logger.info("")
        logger.info("="*70)
        logger.info("Physics Evaluation Complete")
        logger.info("="*70)
        logger.info(f"Overall Physics Score: {overall_score:.6f}")
        logger.info(f"  Boundary Score: {results['boundary_score']:.6f}")
        logger.info(f"  Smoothness Score: {results['smoothness_score']:.6f}")
        logger.info("="*70)
        
        report = evaluator.generate_evaluation_report(results)
        report_path = save_dir / 'data' / 'physics_evaluation_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Physics evaluation report saved: {report_path}")
        
        return results
        
    except Exception as e:
        logger.error("")
        logger.error("="*70)
        logger.error("Physics Evaluation FAILED")
        logger.error("="*70)
        logger.error(f"Error: {e}")
        logger.error("Continuing without physics evaluation...")
        logger.error("="*70)
        return None


# ==============================================================================
# K-Fold Experiment Manager
# ==============================================================================

class KFoldExperimentManager:
    """Manager for k-fold cross-validation M4 experiment."""
    
    def __init__(self, config: M4ViscosityConfig):
        self.config = config
        self.n_folds = config.k_folds
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = config.output_dir / f'run_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results: List[Dict[str, Any]] = []
        self.logger = setup_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def run_all_folds(self) -> None:
        """Run experiments for all folds."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"K-Fold M4 Experiment - Viscosity System ({self.n_folds} folds)")
        self.logger.info("="*70)
        
        total_start = time.time()
        
        low_temp_path = self.config.data_dir / self.config.low_temp_file
        high_temp_path = self.config.data_dir / self.config.high_temp_file
        
        self.logger.info("")
        self.logger.info("Loading data and creating k-fold splits...")
        
        X_low_temp, y_low_temp = load_viscosity_data(low_temp_path)
        X_test, y_test = load_viscosity_data(high_temp_path)
        
        folds = create_k_folds(
            X_low_temp,
            y_low_temp,
            n_splits=self.n_folds,
            random_state=self.config.kfold_random_state
        )
        
        self.logger.info(f"K-fold splits created (n_splits={self.n_folds})")
        for i, fold in enumerate(folds):
            self.logger.info(
                f"  Fold {i}: Train={len(fold['X_train'])}, Val={len(fold['X_val'])}"
            )
        
        for fold_idx, fold_data in enumerate(folds):
            self.logger.info("")
            self.logger.info("="*70)
            self.logger.info(f"Fold {fold_idx} Experiment")
            self.logger.info("="*70)
            
            fold_dir = self.output_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)
            
            X_train = fold_data['X_train']
            y_train = fold_data['y_train']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            
            results = train_m4_pipeline(
                X_train, y_train, X_val, y_val, X_test, y_test,
                self.config, fold_dir
            )
            
            if self.config.save_predictions:
                save_predictions_to_excel(
                    X_train, y_train, results['predictions']['train'],
                    X_val, y_val, results['predictions']['val'],
                    X_test, y_test, results['predictions']['test'],
                    fold_dir,
                    prefix=self.config.excel_prefix
                )
            
            physics_results = run_physics_evaluation(
                results['wrapper'],
                self.config,
                fold_dir
            )
            
            if self.config.save_metrics:
                save_fold_metrics_to_excel(
                    results['metrics'],
                    physics_results,
                    fold_dir / 'excel' / f'{self.config.excel_prefix}metrics.xlsx',
                    self.logger
                )
            
            save_config(
                results['metrics'],
                physics_results,
                self.config,
                fold_dir,
                fold_idx
            )
            
            result_dict = {
                'fold_idx': fold_idx,
                'metrics': results['metrics'],
                'physics_results': physics_results
            }
            self.all_results.append(result_dict)
            
            self.logger.info(f"Fold {fold_idx} completed")
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("Generating summary...")
        self.logger.info("="*70)
        
        self.generate_summary()
        
        total_time = time.time() - total_start
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("All experiments completed!")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
        self.logger.info(f"Results saved to: {self.output_dir}")
    
    def generate_summary(self) -> None:
        """Generate summary across all folds."""
        summary_dir = self.output_dir / 'summary'
        summary_dir.mkdir(exist_ok=True)
        
        metrics_data = []
        
        for result in self.all_results:
            fold_idx = result['fold_idx']
            metrics = result['metrics']
            physics = result.get('physics_results')
            
            row = {
                'Fold': fold_idx,
                'Train_R2': metrics['train_r2'],
                'Val_R2': metrics['val_r2'],
                'Test_R2': metrics['test_r2'],
                'Train_RMSE': metrics['train_rmse'],
                'Val_RMSE': metrics['val_rmse'],
                'Test_RMSE': metrics['test_rmse'],
                'Train_MAE': metrics['train_mae'],
                'Val_MAE': metrics['val_mae'],
                'Test_MAE': metrics['test_mae']
            }
            
            if physics is not None:
                row['Physics_Overall'] = physics.get('overall_score', np.nan)
                row['Physics_Boundary'] = physics.get('boundary_score', np.nan)
                row['Physics_Smoothness'] = physics.get('smoothness_score', np.nan)
            else:
                row['Physics_Overall'] = np.nan
                row['Physics_Boundary'] = np.nan
                row['Physics_Smoothness'] = np.nan
            
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        df.to_excel(summary_dir / 'all_folds_metrics.xlsx', index=False, engine='openpyxl')
        
        summary_stats = df.describe().T
        summary_stats.to_excel(summary_dir / 'summary_statistics.xlsx', engine='openpyxl')
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("K-Fold M4 Experiment Summary - Viscosity System")
        report_lines.append("="*70)
        report_lines.append(f"\nNumber of folds: {self.n_folds}")
        report_lines.append(f"Random state: {self.config.kfold_random_state}")
        report_lines.append("\n" + "-"*70)
        report_lines.append("Test Set Performance (Mean ± Std)")
        report_lines.append("-"*70)
        
        test_r2_mean = df['Test_R2'].mean()
        test_r2_std = df['Test_R2'].std()
        test_rmse_mean = df['Test_RMSE'].mean()
        test_rmse_std = df['Test_RMSE'].std()
        
        report_lines.append(f"Test R²:   {test_r2_mean:.6f} ± {test_r2_std:.6f}")
        report_lines.append(f"Test RMSE: {test_rmse_mean:.6f} ± {test_rmse_std:.6f}")
        
        if not df['Physics_Overall'].isna().all():
            phys_mean = df['Physics_Overall'].mean()
            phys_std = df['Physics_Overall'].std()
            report_lines.append(f"Physics Score: {phys_mean:.6f} ± {phys_std:.6f}")
        
        report_lines.append("\n" + "="*70)
        
        report_text = "\n".join(report_lines)
        
        with open(summary_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        self.logger.info("\n" + report_text)


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main entry point for M4 viscosity experiment."""
    config = M4ViscosityConfig()
    
    logger.info("")
    logger.info("="*70)
    logger.info("Project Paths")
    logger.info("="*70)
    logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"  - Low temp (<40°C): {config.data_dir / config.low_temp_file}")
    logger.info(f"  - High temp (≥40°C): {config.data_dir / config.high_temp_file}")
    logger.info(f"Models directory: {config.models_dir}")
    logger.info(f"  - MCH-HMN: {config.models_dir / config.mch_hmn_model_file}")
    logger.info(f"  - MCH-Dec: {config.models_dir / config.mch_dec_model_file}")
    logger.info(f"  - Dec-HMN: {config.models_dir / config.dec_hmn_model_file}")
    logger.info(f"Output directory: {config.output_dir}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("M4 Viscosity Experiment - Complete Framework")
    logger.info("="*70)
    logger.info(f"Device: {config.device}")
    logger.info(f"K-folds: {config.k_folds}")
    logger.info(f"Random state: {config.kfold_random_state}")
    logger.info("")
    logger.info("M4 Configuration:")
    logger.info(f"  Module 1 - TKMeansLOF: {'Enabled' if config.enable_cleaning else 'Disabled'}")
    logger.info(f"  Module 2 - IADAF:")
    logger.info(f"    - Synthetic samples: {config.n_samples_to_generate}")
    logger.info(f"    - WGAN epochs: {config.iadaf_config['WGAN_EPOCHS']}")
    logger.info(f"    - BO iterations: {config.iadaf_config['BO_N_ITERATIONS']}")
    logger.info(f"  Module 3 - LDPC:")
    logger.info(f"    - Architecture: {config.ldpc_dnn_layers} layers, {config.ldpc_dnn_nodes} nodes")
    logger.info(f"    - Epochs: {config.ldpc_epochs}")
    logger.info(f"    - Learning rate: {config.ldpc_lr}")
    logger.info(f"    - Edge samples: {config.n_edge_samples}, eps={config.edge_weight_eps}")
    logger.info("")
    logger.info("Physics evaluation:")
    logger.info(f"  Boundary lambda: {config.boundary_decay_lambda}")
    logger.info(f"  Smoothness lambda: {config.smoothness_decay_lambda}")
    
    low_temp_path = config.data_dir / config.low_temp_file
    high_temp_path = config.data_dir / config.high_temp_file
    
    if not low_temp_path.exists():
        logger.error(f"Low temp data file not found: {low_temp_path}")
        logger.error("Please check data_dir and low_temp_file in configuration")
        return
    
    if not high_temp_path.exists():
        logger.error(f"High temp data file not found: {high_temp_path}")
        logger.error("Please check data_dir and high_temp_file in configuration")
        return
    
    mch_hmn_path = config.models_dir / config.mch_hmn_model_file
    mch_dec_path = config.models_dir / config.mch_dec_model_file
    dec_hmn_path = config.models_dir / config.dec_hmn_model_file
    
    if not mch_hmn_path.exists():
        logger.error(f"MCH-HMN model not found: {mch_hmn_path}")
        return
    if not mch_dec_path.exists():
        logger.error(f"MCH-Dec model not found: {mch_dec_path}")
        return
    if not dec_hmn_path.exists():
        logger.error(f"Dec-HMN model not found: {dec_hmn_path}")
        return
    
    logger.info("")
    logger.info("All required files validated successfully")
    
    manager = KFoldExperimentManager(config)
    manager.run_all_folds()
    
    logger.info("")
    logger.info("M4 viscosity experiment completed successfully")
    logger.info("")


if __name__ == "__main__":
    main()