"""
================================================================================
Physics Evaluation Utilities for Ternary Solubility Systems
================================================================================

This module provides comprehensive physics-based evaluation tools for assessing
thermodynamic consistency in ternary solubility systems.

Core Features:
    - Boundary consistency evaluation with exponential decay scoring
    - Thermodynamic smoothness evaluation using 2D Laplacian analysis
    - Physical consistency evaluation framework (boundary + smoothness)
    - TSTR (Train on Synthetic, Test on Real) evaluation
    - Configurable logging for debugging

Standardization:
    - Uses BinaryPredictor from binary_predictor.py module
    - Consistent naming: input_boundary_model, output_boundary_model
    - Fully typed with complete docstrings
    - PEP 8 compliant
    - No chemical-specific names (generic component terminology)
    - Uses logging module for output control

================================================================================
"""

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
    
    Examples:
        >>> logger = get_logger(__name__, logging.DEBUG)
        >>> logger.info("Starting evaluation")
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
# Import Binary Predictor Module
# ==============================================================================

models_path = Path(__file__).parent.parent.parent / 'src' / 'models'
sys.path.insert(0, str(models_path))

from binary_predictor import BinaryPredictor


# ==============================================================================
# Neural Network Architecture
# ==============================================================================

class FcBlock(nn.Module):
    """
    Fully connected block with batch normalization and PReLU activation.
    
    Attributes:
        fc: Linear transformation layer.
        bn: Batch normalization layer.
        act: PReLU activation function.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize fully connected block.
        
        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
        """
        super(FcBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through block."""
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DNN(nn.Module):
    """
    Deep neural network with configurable architecture.
    
    Architecture: Input → FcBlock → ... → FcBlock → Output
    
    Attributes:
        fc1: Input layer.
        fcList: List of hidden layers.
        fcn: Output layer.
    """
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        layer_dim: int = 3, 
        node_dim: int = 100
    ):
        """
        Initialize deep neural network.
        
        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            layer_dim: Number of layers (minimum 2).
            node_dim: Number of nodes per hidden layer.
        """
        super(DNN, self).__init__()
        self.fc1 = FcBlock(in_dim, node_dim)
        self.fcList = nn.ModuleList([
            FcBlock(node_dim, node_dim) 
            for _ in range(layer_dim - 2)
        ])
        self.fcn = FcBlock(node_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = self.fc1(x)
        for fc in self.fcList:
            x = fc(x)
        x = self.fcn(x)
        return x


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class PhysicsConfig:
    """
    Configuration for physics evaluation.
    
    Attributes:
        Temperature Range:
            T_min: Minimum temperature (°C).
            T_max: Maximum temperature (°C).
            T_boundary_points: Number of temperature points for boundary evaluation.
        
        Concentration Range:
            W_min: Minimum concentration (%).
            W_max: Maximum concentration (%).
            output_ref_range: Reference range for output boundary normalization.
        
        Grid Resolution:
            grid_resolution_T: Temperature grid resolution.
            grid_resolution_W: Concentration grid resolution.
        
        Boundary Lines:
            boundary_line_width: Width of boundary lines.
            boundary_line_alpha: Transparency of boundary lines.
        
        TSTR Training:
            tstr_epochs: Number of training epochs.
            tstr_lr: Learning rate.
            tstr_batch_size: Batch size.
            tstr_device: Computing device.
            best_model_start_epoch: Epoch to start saving best model.
        
        Model Architecture:
            dnn_layer_dim: Number of DNN layers.
            dnn_node_dim: Number of nodes per layer.
        
        Scoring Parameters:
            boundary_decay_lambda: Decay coefficient for boundary scoring.
            smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        
        Logging:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    T_min: float = -40.0
    T_max: float = 200.0
    T_boundary_points: int = 100
    
    W_min: float = 0.0
    W_max: float = 50.0
    output_ref_range: Tuple[float, float] = (0.0, 80.0)
    
    grid_resolution_T: int = 300
    grid_resolution_W: int = 200
    
    boundary_line_width: float = 3.0
    boundary_line_alpha: float = 0.9
    
    tstr_epochs: int = 1000
    tstr_batch_size: int = 2000
    tstr_lr: float = 0.00831
    tstr_device: str = 'cuda'
    best_model_start_epoch: int = 500
    
    dnn_layer_dim: int = 4
    dnn_node_dim: int = 128
    
    boundary_decay_lambda: float = 5.0
    smoothness_decay_lambda: float = 15.0
    
    log_level: int = logging.INFO


# ==============================================================================
# Boundary NRMSE Calculation
# ==============================================================================

def calculate_boundary_nrmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    physical_max: float
) -> float:
    """
    Calculate normalized root mean square error for boundary predictions.
    
    Args:
        y_pred: Predicted values [N].
        y_true: True values [N].
        physical_max: Maximum physical value for normalization.
    
    Returns:
        NRMSE value (0-1 scale).
    
    Examples:
        >>> nrmse = calculate_boundary_nrmse(pred, true, physical_max=100.0)
        >>> print(f"NRMSE: {nrmse:.4f}")
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nrmse = rmse / (physical_max + 1e-8)
    return float(nrmse)


# ==============================================================================
# Exponential Decay Scoring
# ==============================================================================

def exponential_decay_score(total_error: float, decay_lambda: float = 5.0) -> float:
    """
    Calculate score using exponential decay function.
    
    Formula: S = exp(-λ * ε)
    
    Args:
        total_error: Total accumulated error.
        decay_lambda: Decay coefficient (higher = stricter scoring).
    
    Returns:
        Score in range [0, 1].
    
    Examples:
        >>> score = exponential_decay_score(0.05, decay_lambda=5.0)
        >>> print(f"Score: {score:.4f}")
    """
    return float(np.exp(-decay_lambda * total_error))


# ==============================================================================
# Data Loading and Splitting
# ==============================================================================

def load_ternary_data(
    filepath: str,
    input_cols: List[str] = None,
    output_col: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ternary system data from file.
    
    Args:
        filepath: Path to data file (CSV or Excel).
        input_cols: Names of input columns.
        output_col: Name of output column.
    
    Returns:
        Tuple of (X, y) arrays.
    
    Examples:
        >>> X, y = load_ternary_data('data.csv', ['T', 'W_input'], 'W_output')
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    if input_cols is None:
        input_cols = df.columns[:-1].tolist()
    if output_col is None:
        output_col = df.columns[-1]
    
    X = df[input_cols].values
    y = df[output_col].values
    
    logger.info(f"Loaded data from {filepath}: {len(X)} samples")
    return X, y


def split_data_three_way(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input features [N, D].
        y: Target values [N].
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        random_state: Random seed.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    
    Examples:
        >>> X_tr, y_tr, X_val, y_val, X_te, y_te = split_data_three_way(X, y)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_state
    )
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# Binary Model Loading
# ==============================================================================

def load_binary_model(
    model_path: str,
    model_type: str = 'auto'
) -> BinaryPredictor:
    """
    Load a trained binary predictor model.
    
    Args:
        model_path: Path to saved model file.
        model_type: Type of model ('auto', 'pytorch', 'sklearn').
    
    Returns:
        Loaded BinaryPredictor instance.
    
    Examples:
        >>> model = load_binary_model('models/binary_model.pth')
    """
    model = BinaryPredictor.load(model_path, model_type=model_type)
    logger.info(f"Loaded binary model from {model_path}")
    return model


def load_boundary_models(
    input_model_path: str,
    output_model_path: str
) -> Tuple[BinaryPredictor, BinaryPredictor]:
    """
    Load both input and output boundary models.
    
    Args:
        input_model_path: Path to input component model.
        output_model_path: Path to output component model.
    
    Returns:
        Tuple of (input_boundary_model, output_boundary_model).
    
    Examples:
        >>> input_model, output_model = load_boundary_models(
        ...     'models/input.pth', 'models/output.pth'
        ... )
    """
    input_model = load_binary_model(input_model_path)
    output_model = load_binary_model(output_model_path)
    logger.info("Loaded both boundary models")
    return input_model, output_model


# ==============================================================================
# DNN Boundary Evaluator
# ==============================================================================

class DNNBoundaryEvaluator:
    """
    Evaluator for DNN boundary consistency against binary models.
    
    Attributes:
        input_boundary_model: Binary predictor for input component.
        output_boundary_model: Binary predictor for output component.
        config: Physics configuration.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        input_boundary_model: BinaryPredictor,
        output_boundary_model: BinaryPredictor,
        config: PhysicsConfig = None
    ):
        """
        Initialize boundary evaluator.
        
        Args:
            input_boundary_model: Model for input component boundary.
            output_boundary_model: Model for output component boundary.
            config: Physics configuration (uses defaults if None).
        """
        if config is None:
            config = PhysicsConfig()
        
        self.config = config
        self.input_boundary_model = input_boundary_model
        self.output_boundary_model = output_boundary_model
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info("DNNBoundaryEvaluator initialized")
    
    def evaluate_boundaries(
        self,
        dnn_predict_fn: Callable[[np.ndarray], np.ndarray],
        T_points: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Evaluate DNN predictions against binary model boundaries.
        
        Args:
            dnn_predict_fn: Function that predicts output given [T, W_input].
            T_points: Temperature points for evaluation (uses config default if None).
        
        Returns:
            Dictionary containing:
                - 'input_nrmse': NRMSE for input boundary
                - 'output_nrmse': NRMSE for output boundary
                - 'input_boundary_true': True input boundary values
                - 'input_boundary_pred': Predicted input boundary values
                - 'output_boundary_true': True output boundary values
                - 'output_boundary_pred': Predicted output boundary values
                - 'T_points': Temperature points used
        
        Examples:
            >>> results = evaluator.evaluate_boundaries(dnn_model.predict)
            >>> print(f"Input NRMSE: {results['input_nrmse']:.4f}")
        """
        if T_points is None:
            T_points = np.linspace(
                self.config.T_min,
                self.config.T_max,
                self.config.T_boundary_points
            ).reshape(-1, 1)
        
        self.logger.debug(f"Evaluating boundaries at {len(T_points)} temperature points")
        
        input_boundary_true = self.input_boundary_model.predict(T_points).flatten()
        X_input_boundary = np.column_stack([T_points.flatten(), input_boundary_true])
        input_boundary_pred = dnn_predict_fn(X_input_boundary)
        if input_boundary_pred.ndim > 1:
            input_boundary_pred = input_boundary_pred.flatten()
        
        output_boundary_true = self.output_boundary_model.predict(T_points).flatten()
        X_output_boundary = np.column_stack([T_points.flatten(), np.zeros_like(T_points.flatten())])
        output_boundary_pred = dnn_predict_fn(X_output_boundary)
        if output_boundary_pred.ndim > 1:
            output_boundary_pred = output_boundary_pred.flatten()
        
        input_nrmse = calculate_boundary_nrmse(
            input_boundary_pred,
            np.zeros_like(input_boundary_pred),
            self.config.output_ref_range[1]
        )
        
        output_nrmse = calculate_boundary_nrmse(
            output_boundary_pred,
            output_boundary_true,
            self.config.output_ref_range[1]
        )
        
        self.logger.info(f"Boundary evaluation - Input NRMSE: {input_nrmse:.4f}, "
                        f"Output NRMSE: {output_nrmse:.4f}")
        
        return {
            'input_nrmse': float(input_nrmse),
            'output_nrmse': float(output_nrmse),
            'input_boundary_true': np.zeros_like(input_boundary_pred),
            'input_boundary_pred': input_boundary_pred,
            'output_boundary_true': output_boundary_true,
            'output_boundary_pred': output_boundary_pred,
            'T_points': T_points.flatten()
        }


# ==============================================================================
# Physical Consistency Evaluator
# ==============================================================================

class PhysicalConsistencyEvaluator:
    """
    Evaluator for physical consistency using dual-pillar framework.
    
    Combines boundary consistency and thermodynamic smoothness evaluations.
    
    Attributes:
        boundary_evaluator: DNNBoundaryEvaluator instance.
        config: Physics configuration.
        boundary_decay_lambda: Decay coefficient for boundary scoring.
        smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        input_boundary_model: BinaryPredictor,
        output_boundary_model: BinaryPredictor,
        config: PhysicsConfig = None,
        boundary_decay_lambda: float = None,
        smoothness_decay_lambda: float = None
    ):
        """
        Initialize physical consistency evaluator.
        
        Args:
            input_boundary_model: Model for input component boundary.
            output_boundary_model: Model for output component boundary.
            config: Physics configuration (uses defaults if None).
            boundary_decay_lambda: Custom decay for boundary (uses config if None).
            smoothness_decay_lambda: Custom decay for smoothness (uses config if None).
        """
        if config is None:
            config = PhysicsConfig()
        
        self.config = config
        self.boundary_evaluator = DNNBoundaryEvaluator(
            input_boundary_model,
            output_boundary_model,
            config
        )
        
        self.boundary_decay_lambda = boundary_decay_lambda or config.boundary_decay_lambda
        self.smoothness_decay_lambda = smoothness_decay_lambda or config.smoothness_decay_lambda
        
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        self.logger.info("PhysicalConsistencyEvaluator initialized")
    
    def evaluate_smoothness(
        self,
        predicted_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate thermodynamic smoothness using 2D Laplacian.
        
        Args:
            predicted_data: Array of [T, W_input, W_output].
        
        Returns:
            Dictionary containing:
                - 'eta': Smoothness metric (99th percentile of normalized Laplacian)
                - 'laplacian_p99': 99th percentile of absolute Laplacian
                - 'data_range': Range of W_output values
                - 'quality_level': Quality assessment
        
        Examples:
            >>> results = evaluator.evaluate_smoothness(predictions)
            >>> print(f"η = {results['eta']:.4f}")
        """
        self.logger.debug("Evaluating thermodynamic smoothness")
        
        T = predicted_data[:, 0]
        W_input = predicted_data[:, 1]
        W_output = predicted_data[:, 2]
        
        T_grid = np.linspace(T.min(), T.max(), self.config.grid_resolution_T)
        W_grid = np.linspace(W_input.min(), W_input.max(), self.config.grid_resolution_W)
        T_mesh, W_mesh = np.meshgrid(T_grid, W_grid)
        
        W_output_grid = griddata(
            (T, W_input),
            W_output,
            (T_mesh, W_mesh),
            method='cubic'
        )
        
        dT = T_grid[1] - T_grid[0]
        dW = W_grid[1] - W_grid[0]
        
        laplacian = (
            np.roll(W_output_grid, 1, axis=0) +
            np.roll(W_output_grid, -1, axis=0) +
            np.roll(W_output_grid, 1, axis=1) +
            np.roll(W_output_grid, -1, axis=1) -
            4 * W_output_grid
        ) / (dT * dW)
        
        laplacian = laplacian[1:-1, 1:-1]
        
        laplacian_abs = np.abs(laplacian[~np.isnan(laplacian)])
        laplacian_p99 = np.percentile(laplacian_abs, 99)
        data_range = W_output.max() - W_output.min()
        eta = laplacian_p99 / (data_range + 1e-8)
        
        if eta < 0.02:
            quality_level = 'Excellent'
        elif eta < 0.04:
            quality_level = 'Good'
        elif eta < 0.08:
            quality_level = 'Acceptable'
        elif eta < 0.15:
            quality_level = 'Poor'
        else:
            quality_level = 'Failed'
        
        self.logger.info(f"Smoothness evaluation - η: {eta:.4f} ({quality_level})")
        
        return {
            'eta': float(eta),
            'laplacian_p99': float(laplacian_p99),
            'data_range': float(data_range),
            'quality_level': quality_level
        }
    
    def evaluate_full(
        self,
        dnn_model: nn.Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        predicted_data: np.ndarray,
        device: str = 'cuda'
    ) -> Tuple[float, Dict]:
        """
        Full evaluation combining boundary and smoothness.
        
        Args:
            dnn_model: Trained DNN model.
            x_scaler: Input scaler.
            y_scaler: Output scaler.
            predicted_data: Array of [T, W_input, W_output].
            device: Computing device.
        
        Returns:
            Tuple of (overall_score, detailed_results_dict).
        
        Examples:
            >>> score, results = evaluator.evaluate_full(
            ...     model, scaler_x, scaler_y, predictions
            ... )
            >>> print(f"Overall score: {score:.4f}")
        """
        self.logger.info("Starting full physics evaluation")
        
        def predict_fn(X: np.ndarray) -> np.ndarray:
            dnn_model.eval()
            with torch.no_grad():
                X_scaled = x_scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                y_pred_scaled = dnn_model(X_tensor).cpu().numpy()
                y_pred = y_scaler.inverse_transform(y_pred_scaled)
                return y_pred.flatten()
        
        boundary_results = self.boundary_evaluator.evaluate_boundaries(predict_fn)
        boundary_error = (
            boundary_results['input_nrmse'] +
            boundary_results['output_nrmse']
        )
        boundary_score = exponential_decay_score(
            boundary_error,
            self.boundary_decay_lambda
        )
        
        smoothness_results = self.evaluate_smoothness(predicted_data)
        smoothness_score = exponential_decay_score(
            smoothness_results['eta'],
            self.smoothness_decay_lambda
        )
        
        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score
        
        self.logger.info(f"Boundary score: {boundary_score:.6f}")
        self.logger.info(f"Smoothness score: {smoothness_score:.6f}")
        self.logger.info(f"Overall score: {overall_score:.6f}")
        
        results = {
            'overall_score': float(overall_score),
            'boundary_score': float(boundary_score),
            'smoothness_score': float(smoothness_score),
            'boundary_error': float(boundary_error),
            'smoothness_eta': float(smoothness_results['eta']),
            'boundary_details': boundary_results,
            'smoothness_details': smoothness_results
        }
        
        return overall_score, results
    
    def evaluate_with_predictor(
        self,
        predict_fn: Callable,
        predicted_data: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Simplified evaluation using a prediction function.
        
        Args:
            predict_fn: Function that takes [T, W_input] and returns W_output.
            predicted_data: Array of [T, W_input, W_output].
        
        Returns:
            Tuple of (overall_score, detailed_results_dict).
        """
        boundary_results = self.boundary_evaluator.evaluate_boundaries(predict_fn)
        boundary_error = (
            boundary_results['input_nrmse'] +
            boundary_results['output_nrmse']
        )
        boundary_score = exponential_decay_score(
            boundary_error,
            self.boundary_decay_lambda
        )
        
        smoothness_results = self.evaluate_smoothness(predicted_data)
        smoothness_score = exponential_decay_score(
            smoothness_results['eta'],
            self.smoothness_decay_lambda
        )
        
        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score
        
        results = {
            'overall_score': float(overall_score),
            'boundary_score': float(boundary_score),
            'smoothness_score': float(smoothness_score),
            'boundary_error': float(boundary_error),
            'smoothness_eta': float(smoothness_results['eta']),
            'boundary_details': boundary_results,
            'smoothness_details': smoothness_results
        }
        
        return overall_score, results
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """
        Generate a formatted evaluation report.
        
        Args:
            results: Results dictionary from evaluate_full or evaluate_with_predictor.
        
        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 80)
        report.append("PHYSICS-BASED EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 80)
        report.append(f"Overall Score:      {results['overall_score']:.6f}")
        report.append("")
        
        report.append("BOUNDARY CONSISTENCY")
        report.append("-" * 80)
        report.append(f"Boundary Score:     {results['boundary_score']:.6f}")
        report.append(f"Total Boundary Error: {results['boundary_error']:.4f}")
        report.append(f"  - Input NRMSE:    {results['boundary_details']['input_nrmse']:.4f}")
        report.append(f"  - Output NRMSE:   {results['boundary_details']['output_nrmse']:.4f}")
        report.append("")
        
        report.append("THERMODYNAMIC SMOOTHNESS")
        report.append("-" * 80)
        report.append(f"Smoothness Score:   {results['smoothness_score']:.6f}")
        report.append(f"η (Normalized Curvature): {results['smoothness_eta']:.4f}")
        report.append(f"Quality Level:      {results['smoothness_details']['quality_level']}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# ==============================================================================
# TSTR Evaluator
# ==============================================================================

class TSTREvaluator:
    """
    Train on Synthetic, Test on Real (TSTR) evaluator.
    
    Attributes:
        X_val: Validation features.
        y_val: Validation targets.
        X_test: Test features.
        y_test: Test targets.
        X_train: Training features (optional).
        y_train: Training targets (optional).
        config: Physics configuration.
        logger: Logger instance.
    """
    
    def __init__(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        config: PhysicsConfig = None
    ):
        """
        Initialize TSTR evaluator.
        
        Args:
            X_val: Validation features [N_val, D].
            y_val: Validation targets [N_val].
            X_test: Test features [N_test, D].
            y_test: Test targets [N_test].
            X_train: Training features [N_train, D] (optional).
            y_train: Training targets [N_train] (optional).
            config: Physics configuration (uses defaults if None).
        """
        if config is None:
            config = PhysicsConfig()
        
        self.config = config
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.logger = get_logger(self.__class__.__name__, config.log_level)
        
        self.logger.info("TSTREvaluator initialized")
        self.logger.debug(f"Validation set size: {len(X_val)}")
        self.logger.debug(f"Test set size: {len(X_test)}")
    
    def evaluate(
        self,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        epochs: int = None,
        random_seed: int = 42,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train on synthetic data and test on real data.
        
        Args:
            X_syn: Synthetic training features [N_syn, D].
            y_syn: Synthetic training targets [N_syn].
            epochs: Number of training epochs (uses config default if None).
            random_seed: Random seed for reproducibility.
            verbose: Whether to print training progress.
        
        Returns:
            Dictionary containing:
                - 'metrics': R², RMSE, MAE on train/val/test sets
                - 'history': Training history (train/val/test R² per epoch)
                - 'model': Best trained model
                - 'x_scaler': Fitted input scaler
                - 'y_scaler': Fitted output scaler
                - 'predictions': Predictions on train/val/test sets
                - 'true_values': True values for train/val/test sets
                - 'inputs': Input features for train/val/test sets
                - 'best_epoch': Epoch with best validation R²
                - 'best_val_r2': Best validation R²
        
        Examples:
            >>> result = evaluator.evaluate(X_syn, y_syn, epochs=1000, verbose=True)
            >>> print(f"Test R²: {result['metrics']['test_r2']:.4f}")
            >>> print(f"Best epoch: {result['best_epoch']}")
        """
        if epochs is None:
            epochs = self.config.tstr_epochs
        
        self.logger.info(f"Starting TSTR evaluation with {len(X_syn)} synthetic samples")
        self.logger.info(f"Training for {epochs} epochs")
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        device = torch.device(self.config.tstr_device if torch.cuda.is_available() else 'cpu')
        self.logger.debug(f"Using device: {device}")
        
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X_syn_scaled = x_scaler.fit_transform(X_syn)
        y_syn_scaled = y_scaler.fit_transform(y_syn.reshape(-1, 1)).flatten()
        
        X_val_scaled = x_scaler.transform(self.X_val)
        X_test_scaled = x_scaler.transform(self.X_test)
        
        g = torch.Generator()
        g.manual_seed(random_seed)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_syn_scaled),
            torch.FloatTensor(y_syn_scaled)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.tstr_batch_size,
            shuffle=True,
            generator=g
        )
        
        in_dim = X_syn.shape[1]
        model = DNN(
            in_dim=in_dim,
            out_dim=1,
            layer_dim=self.config.dnn_layer_dim,
            node_dim=self.config.dnn_node_dim
        ).to(device)
        
        self.logger.debug(f"Model architecture: {self.config.dnn_layer_dim} layers, "
                         f"{self.config.dnn_node_dim} nodes")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.tstr_lr)
        criterion = nn.MSELoss()
        
        best_val_r2 = -float('inf')
        best_model_state = None
        best_epoch = 0
        
        history = {
            'train_r2': [],
            'val_r2': [],
            'test_r2': []
        }
        
        pbar = tqdm(range(epochs), desc="TSTR Training") if verbose else range(epochs)
        
        for epoch in pbar:
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                X_train_tensor = torch.FloatTensor(X_syn_scaled).to(device)
                y_train_pred_scaled = model(X_train_tensor).cpu().numpy()
                y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).flatten()
                train_r2 = r2_score(y_syn, y_train_pred)
                
                X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
                y_val_pred_scaled = model(X_val_tensor).cpu().numpy()
                y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled).flatten()
                val_r2 = r2_score(self.y_val, y_val_pred)
                
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                y_test_pred_scaled = model(X_test_tensor).cpu().numpy()
                y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled).flatten()
                test_r2 = r2_score(self.y_test, y_test_pred)
            
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)
            history['test_r2'].append(test_r2)
            
            if val_r2 > best_val_r2 and epoch >= self.config.best_model_start_epoch:
                best_val_r2 = val_r2
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'train_r2': f'{train_r2:.4f}',
                    'val_r2': f'{val_r2:.4f}',
                    'test_r2': f'{test_r2:.4f}'
                })
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self.logger.info(f"Loaded best model from epoch {best_epoch} "
                           f"(Val R² = {best_val_r2:.6f})")
        else:
            self.logger.warning(f"No best model saved (all epochs < "
                              f"{self.config.best_model_start_epoch})")
        
        model.eval()
        with torch.no_grad():
            X_train_tensor = torch.FloatTensor(X_syn_scaled).to(device)
            y_train_pred_final = model(X_train_tensor).cpu().numpy()
            y_train_pred_final = y_scaler.inverse_transform(y_train_pred_final).flatten()
            
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_pred_final = model(X_val_tensor).cpu().numpy()
            y_val_pred_final = y_scaler.inverse_transform(y_val_pred_final).flatten()
            
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_test_pred_final = model(X_test_tensor).cpu().numpy()
            y_test_pred_final = y_scaler.inverse_transform(y_test_pred_final).flatten()
        
        metrics_result = {
            'train_r2': float(r2_score(y_syn, y_train_pred_final)),
            'train_rmse': float(np.sqrt(mean_squared_error(y_syn, y_train_pred_final))),
            'train_mae': float(mean_absolute_error(y_syn, y_train_pred_final)),
            'val_r2': float(r2_score(self.y_val, y_val_pred_final)),
            'val_rmse': float(np.sqrt(mean_squared_error(self.y_val, y_val_pred_final))),
            'val_mae': float(mean_absolute_error(self.y_val, y_val_pred_final)),
            'test_r2': float(r2_score(self.y_test, y_test_pred_final)),
            'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, y_test_pred_final))),
            'test_mae': float(mean_absolute_error(self.y_test, y_test_pred_final))
        }
        
        print("\n" + "=" * 80)
        print("TSTR EVALUATION RESULTS")
        print("=" * 80)
        print(f"\nTraining Set Metrics:")
        print(f"  R²:   {metrics_result['train_r2']:.6f}")
        print(f"  RMSE: {metrics_result['train_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['train_mae']:.6f}")
        print(f"\nValidation Set Metrics:")
        print(f"  R²:   {metrics_result['val_r2']:.6f}")
        print(f"  RMSE: {metrics_result['val_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['val_mae']:.6f}")
        print(f"\nTest Set Metrics:")
        print(f"  R²:   {metrics_result['test_r2']:.6f}")
        print(f"  RMSE: {metrics_result['test_rmse']:.6f}")
        print(f"  MAE:  {metrics_result['test_mae']:.6f}")
        if best_model_state is not None:
            print(f"\nBest Model Information:")
            print(f"  Epoch: {best_epoch}")
            print(f"  Validation R²: {best_val_r2:.6f}")
        print("=" * 80 + "\n")
        
        self.logger.info("TSTR evaluation complete")
        
        results = {
            'metrics': metrics_result,
            'history': history,
            'model': model,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'predictions': {
                'train': y_train_pred_final,
                'val': y_val_pred_final,
                'test': y_test_pred_final
            },
            'true_values': {
                'train': y_syn,
                'val': self.y_val,
                'test': self.y_test
            },
            'inputs': {
                'train': X_syn,
                'val': self.X_val,
                'test': self.X_test
            },
            'n_synthetic': len(X_syn),
            'epochs': epochs,
            'best_epoch': best_epoch,
            'best_val_r2': float(best_val_r2)
        }
        
        return results
    
    def evaluate_with_synthetic(
        self,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        epochs: int = None
    ) -> Dict[str, Any]:
        """
        Simplified evaluation interface (alias for evaluate).
        
        Args:
            X_syn: Synthetic training features.
            y_syn: Synthetic training targets.
            epochs: Number of training epochs.
        
        Returns:
            Results dictionary (same as evaluate).
        """
        return self.evaluate(X_syn, y_syn, epochs=epochs, verbose=False)
    
    def save_predictions_to_excel(
        self,
        result: Dict,
        filepath: str
    ):
        """
        Save predictions to Excel file.
        
        Args:
            result: Results dictionary from evaluate.
            filepath: Output Excel file path.
        
        Examples:
            >>> result = evaluator.evaluate(X_syn, y_syn)
            >>> evaluator.save_predictions_to_excel(result, 'predictions.xlsx')
        """
        val_df = pd.DataFrame({
            'Input_T': result['inputs']['val'][:, 0],
            'Input_W': result['inputs']['val'][:, 1],
            'True': self.y_val,
            'Predicted': result['predictions']['val'],
            'Error': self.y_val - result['predictions']['val']
        })
        
        test_df = pd.DataFrame({
            'Input_T': result['inputs']['test'][:, 0],
            'Input_W': result['inputs']['test'][:, 1],
            'True': self.y_test,
            'Predicted': result['predictions']['test'],
            'Error': self.y_test - result['predictions']['test']
        })
        
        train_df = pd.DataFrame({
            'Input_T': result['inputs']['train'][:, 0],
            'Input_W': result['inputs']['train'][:, 1],
            'True': result['true_values']['train'],
            'Predicted': result['predictions']['train'],
            'Error': result['true_values']['train'] - result['predictions']['train']
        })
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            val_df.to_excel(writer, sheet_name='Validation', index=False)
            test_df.to_excel(writer, sheet_name='Test', index=False)
            train_df.to_excel(writer, sheet_name='Training', index=False)
        
        self.logger.info(f"Predictions saved to: {filepath}")
    
    def save_metrics_to_excel(
        self,
        tstr_result: Dict,
        physics_results: Dict,
        filepath: str
    ):
        """
        Save evaluation metrics to Excel file.
        
        Args:
            tstr_result: Results from TSTR evaluation.
            physics_results: Results from physical consistency evaluation.
            filepath: Output Excel file path.
        
        Examples:
            >>> tstr_result = evaluator.evaluate(X_syn, y_syn)
            >>> score, physics_results = consistency_evaluator.evaluate_full(...)
            >>> evaluator.save_metrics_to_excel(tstr_result, physics_results, 'metrics.xlsx')
        """
        metrics_data = {
            'Metric': [
                'Training R²',
                'Training RMSE',
                'Training MAE',
                'Validation R²',
                'Validation RMSE',
                'Validation MAE',
                'Test R²',
                'Test RMSE',
                'Test MAE',
                'Best Model Epoch',
                'Best Validation R²',
                'Boundary Score',
                'Smoothness Score',
                'Overall Physics Score',
                'Number of Synthetic Samples',
                'Training Epochs'
            ],
            'Value': [
                tstr_result['metrics']['train_r2'],
                tstr_result['metrics']['train_rmse'],
                tstr_result['metrics']['train_mae'],
                tstr_result['metrics']['val_r2'],
                tstr_result['metrics']['val_rmse'],
                tstr_result['metrics']['val_mae'],
                tstr_result['metrics']['test_r2'],
                tstr_result['metrics']['test_rmse'],
                tstr_result['metrics']['test_mae'],
                tstr_result['best_epoch'],
                tstr_result['best_val_r2'],
                physics_results['boundary_score'],
                physics_results['smoothness_score'],
                physics_results['overall_score'],
                tstr_result['n_synthetic'],
                tstr_result['epochs']
            ]
        }
        
        df = pd.DataFrame(metrics_data)
        df.to_excel(filepath, index=False)
        
        self.logger.info(f"Metrics saved to: {filepath}")


# ==============================================================================
# Convenience Function
# ==============================================================================

def evaluate_dnn_phase_diagram(
    predicted_data_array: np.ndarray,
    input_boundary_model: BinaryPredictor,
    output_boundary_model: BinaryPredictor,
    dnn_model: nn.Module = None,
    x_scaler: StandardScaler = None,
    y_scaler: StandardScaler = None,
    predict_fn: Callable = None,
    device: str = 'cuda',
    save_report: bool = True,
    report_path: str = 'phase_diagram_evaluation_report.txt',
    boundary_decay_lambda: float = 5.0,
    smoothness_decay_lambda: float = 15.0,
    log_level: int = logging.INFO
) -> Tuple[float, Dict]:
    """
    High-level convenience function for phase diagram evaluation.
    
    Evaluates a DNN-generated phase diagram using the dual-pillar framework
    (boundary consistency + thermodynamic smoothness).
    
    Args:
        predicted_data_array: Predicted phase diagram data [T, W_input, W_output].
        input_boundary_model: Binary model for input component.
        output_boundary_model: Binary model for output component.
        dnn_model: DNN model (required if predict_fn not provided).
        x_scaler: Input scaler (required if dnn_model provided).
        y_scaler: Output scaler (required if dnn_model provided).
        predict_fn: Prediction function (alternative to dnn_model).
        device: Computing device.
        save_report: Whether to save evaluation report.
        report_path: Path to save report.
        boundary_decay_lambda: Decay coefficient for boundary scoring.
        smoothness_decay_lambda: Decay coefficient for smoothness scoring.
        log_level: Logging level.
    
    Returns:
        Tuple of (overall_score, detailed_results_dict).
    
    Examples:
        >>> score, results = evaluate_dnn_phase_diagram(
        ...     predictions,
        ...     input_boundary_model=input_model,
        ...     output_boundary_model=output_model,
        ...     dnn_model=model,
        ...     x_scaler=scaler_x,
        ...     y_scaler=scaler_y
        ... )
        >>> print(f"Physics score: {score:.6f}")
    """
    eval_logger = get_logger(__name__, log_level)
    eval_logger.info("Starting phase diagram evaluation")
    
    config = PhysicsConfig(log_level=log_level)
    
    evaluator = PhysicalConsistencyEvaluator(
        input_boundary_model=input_boundary_model,
        output_boundary_model=output_boundary_model,
        config=config,
        boundary_decay_lambda=boundary_decay_lambda,
        smoothness_decay_lambda=smoothness_decay_lambda
    )
    
    if dnn_model is not None and x_scaler is not None and y_scaler is not None:
        score, results = evaluator.evaluate_full(
            dnn_model, x_scaler, y_scaler,
            predicted_data_array, device
        )
    elif predict_fn is not None:
        score, results = evaluator.evaluate_with_predictor(
            predict_fn, predicted_data_array
        )
    else:
        raise ValueError("Must provide either (dnn_model, x_scaler, y_scaler) or predict_fn")
    
    if save_report:
        report = evaluator.generate_evaluation_report(results)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        eval_logger.info(f"Evaluation report saved: {report_path}")
    
    return score, results


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    'PhysicsConfig',
    'get_logger',
    'load_binary_model',
    'load_boundary_models',
    'load_ternary_data',
    'split_data_three_way',
    'DNN',
    'FcBlock',
    'DNNBoundaryEvaluator',
    'PhysicalConsistencyEvaluator',
    'TSTREvaluator',
    'evaluate_dnn_phase_diagram',
    'calculate_boundary_nrmse',
    'exponential_decay_score',
]