"""
================================================================================
Physics Evaluation Utilities for Ternary Viscosity Systems
================================================================================

Features:
    1. Boundary consistency evaluation (3 boundaries: MCH=0, Dec=0, HMN=0)
    2. Thermodynamic smoothness evaluation (4D Laplacian + P99 quantile)
    3. Dual-pillar comprehensive evaluation framework
    4. Seamless integration with PARL K-fold cross-validation

Theoretical Foundation:
    - Boundary consistency: Gibbs-Duhem theorem (ternary system boundary degeneracy)
    - Thermodynamic smoothness: Cahn-Hilliard equilibrium theory (chemical potential uniformity)

Standardization Changes:
    - English docstrings and comments (PEP 257)
    - Logging module instead of print statements
    - Complete type hints (PEP 484)
    - Google-style docstrings
    - All visualization code removed
    - No matplotlib dependencies
    - No global state modifications
    - 100% backward compatible interfaces


================================================================================
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
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


# Initialize module-level logger
logger = get_logger(__name__)


# ==============================================================================
# Helper Functions
# ==============================================================================

def calculate_boundary_nrmse(
    y_pred: np.ndarray, 
    y_true: np.ndarray, 
    physical_max: float
) -> float:
    """
    Calculate normalized RMSE for boundary evaluation.
    
    Args:
        y_pred: Predicted values.
        y_true: True values.
        physical_max: Physical maximum value for normalization.
    
    Returns:
        Normalized RMSE value.
    
    Examples:
        >>> nrmse = calculate_boundary_nrmse(predictions, targets, 100.0)
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    nrmse = rmse / (physical_max + 1e-8)
    return float(nrmse)


def exponential_decay_score(total_error: float, decay_lambda: float = 5.0) -> float:
    """
    Exponential decay scoring function.
    
    Formula: S = exp(-λ * ε)
    
    Args:
        total_error: Cumulative error.
        decay_lambda: Decay coefficient (default: 5.0).
    
    Returns:
        Score between 0 and 1, where 1 is perfect.
    
    Examples:
        >>> score = exponential_decay_score(0.1, decay_lambda=5.0)
        >>> print(f"Score: {score:.4f}")  # Score: 0.6065
    """
    return float(np.exp(-decay_lambda * total_error))


# ==============================================================================
# Boundary Consistency Evaluator
# ==============================================================================

class ViscosityBoundaryEvaluator:
    """
    Viscosity system boundary consistency evaluator (three boundaries).
    
    Evaluates model predictions against binary system boundaries using
    teacher models for MCH=0, Dec=0, and HMN=0 edges.
    
    Attributes:
        teacher_mch_hmn: Binary model for Dec=0 boundary (MCH-HMN system).
        teacher_dec_hmn: Binary model for MCH=0 boundary (Dec-HMN system).
        teacher_mch_dec: Binary model for HMN=0 boundary (MCH-Dec system).
        temp_range: Temperature range tuple (min, max).
        pressure_range: Pressure range tuple (min, max).
        decay_lambda: Decay coefficient for boundary scoring.
        n_samples: Number of sampling points per boundary.
        logger: Logger instance for this evaluator.
    
    Examples:
        >>> evaluator = ViscosityBoundaryEvaluator(
        ...     teacher_mch_hmn=model1,
        ...     teacher_dec_hmn=model2,
        ...     teacher_mch_dec=model3,
        ...     temp_range=(20.0, 80.0),
        ...     pressure_range=(1e5, 1e8)
        ... )
        >>> results = evaluator.evaluate_parl_boundary(trainer)
    """
    
    def __init__(
        self, 
        teacher_mch_hmn: Any,
        teacher_dec_hmn: Any,
        teacher_mch_dec: Any,
        temp_range: Tuple[float, float] = (20.0, 80.0),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        decay_lambda: float = 5.0,
        n_samples: int = 100,
        log_level: int = logging.INFO
    ):
        """
        Initialize viscosity boundary evaluator.
        
        Args:
            teacher_mch_hmn: Dec=0 boundary teacher model (MCH-HMN binary system).
            teacher_dec_hmn: MCH=0 boundary teacher model (Dec-HMN binary system).
            teacher_mch_dec: HMN=0 boundary teacher model (MCH-Dec binary system).
            temp_range: Temperature range (min, max) in Celsius.
            pressure_range: Pressure range (min, max) in Pa.
            decay_lambda: Boundary consistency decay coefficient.
            n_samples: Number of sampling points per boundary.
            log_level: Logging level.
        """
        self.teacher_mch_hmn = teacher_mch_hmn
        self.teacher_dec_hmn = teacher_dec_hmn
        self.teacher_mch_dec = teacher_mch_dec
        self.temp_range = temp_range
        self.pressure_range = pressure_range
        self.decay_lambda = decay_lambda
        self.n_samples = n_samples
        self.logger = get_logger(self.__class__.__name__, log_level)
        
        # Generate boundary test points
        self._generate_boundary_test_points()
    
    def _generate_boundary_test_points(self) -> None:
        """
        Generate test points for three boundaries (using teacher models for ground truth).
        
        Creates sampling grids for:
            - MCH=0 boundary (Dec-HMN binary system)
            - Dec=0 boundary (MCH-HMN binary system)
            - HMN=0 boundary (MCH-Dec binary system)
        """
        self.logger.info(
            f"Generating boundary test points ({self.n_samples} points per boundary)..."
        )
        
        # Temperature and pressure grid
        T_test = np.linspace(*self.temp_range, self.n_samples)
        P_test = np.linspace(*self.pressure_range, self.n_samples)
        T_grid, P_grid = np.meshgrid(T_test, P_test)
        T_flat = T_grid.flatten()
        P_flat = P_grid.flatten()
        n_tp = len(T_flat)
        
        # ===== Boundary 1: MCH = 0 (Dec-HMN binary system) =====
        self.logger.debug("Generating MCH=0 boundary...")
        Dec_samples = np.linspace(0, 100, self.n_samples)
        
        self.boundary_mch_zero_X = []
        for i in range(n_tp):
            for dec_val in Dec_samples:
                # [T, P, MCH, Dec]
                self.boundary_mch_zero_X.append([T_flat[i], P_flat[i], 0.0, dec_val])
        
        self.boundary_mch_zero_X = np.array(self.boundary_mch_zero_X)
        
        # Teacher model generates ground truth
        self.boundary_mch_zero_y_true = self.teacher_dec_hmn.predict_physical(
            self.boundary_mch_zero_X
        ).flatten()
        
        self.logger.debug(f"MCH=0 boundary: {len(self.boundary_mch_zero_X)} points")
        
        # ===== Boundary 2: Dec = 0 (MCH-HMN binary system) =====
        self.logger.debug("Generating Dec=0 boundary...")
        MCH_samples = np.linspace(0, 100, self.n_samples)
        
        self.boundary_dec_zero_X = []
        for i in range(n_tp):
            for mch_val in MCH_samples:
                # [T, P, MCH, Dec]
                self.boundary_dec_zero_X.append([T_flat[i], P_flat[i], mch_val, 0.0])
        
        self.boundary_dec_zero_X = np.array(self.boundary_dec_zero_X)
        
        # Teacher model generates ground truth
        self.boundary_dec_zero_y_true = self.teacher_mch_hmn.predict_physical(
            self.boundary_dec_zero_X
        ).flatten()
        
        self.logger.debug(f"Dec=0 boundary: {len(self.boundary_dec_zero_X)} points")
        
        # ===== Boundary 3: HMN = 0 (MCH-Dec binary system, MCH+Dec=100) =====
        self.logger.debug("Generating HMN=0 boundary...")
        MCH_samples_hmn = np.linspace(0, 100, self.n_samples)
        
        self.boundary_hmn_zero_X = []
        for i in range(n_tp):
            for mch_val in MCH_samples_hmn:
                dec_val = 100.0 - mch_val
                # [T, P, MCH, Dec]
                self.boundary_hmn_zero_X.append([T_flat[i], P_flat[i], mch_val, dec_val])
        
        self.boundary_hmn_zero_X = np.array(self.boundary_hmn_zero_X)
        
        # Teacher model generates ground truth
        self.boundary_hmn_zero_y_true = self.teacher_mch_dec.predict_physical(
            self.boundary_hmn_zero_X
        ).flatten()
        
        self.logger.debug(f"HMN=0 boundary: {len(self.boundary_hmn_zero_X)} points")
        self.logger.info("Boundary test points generation complete")
    
    def evaluate_parl_boundary(self, trainer: Any) -> Dict[str, Any]:
        """
        Evaluate boundary consistency for PARL trainer.
        
        Computes predictions at three boundary edges and compares against
        binary teacher model predictions.
        
        Args:
            trainer: PARL trainer object with predict() method.
        
        Returns:
            Dictionary containing:
                - Individual boundary metrics (R², RMSE, MAE, NRMSE)
                - Combined boundary score
                - Physical maximum values
                - Detailed predictions
        
        Examples:
            >>> results = evaluator.evaluate_parl_boundary(trainer)
            >>> print(f"Boundary score: {results['combined']['boundary_score']:.6f}")
        """
        self.logger.info("="*70)
        self.logger.info("Boundary Consistency Evaluation")
        self.logger.info("="*70)
        
        results = {}
        
        # ===== Evaluate MCH=0 Boundary =====
        self.logger.info("Evaluating MCH=0 boundary...")
        
        y_pred_mch_zero = trainer.predict(
            self.boundary_mch_zero_X,
            return_original_scale=True
        ).flatten()
        
        y_true_mch_zero = self.boundary_mch_zero_y_true
        
        results['mch_zero_boundary'] = {
            'r2': r2_score(y_true_mch_zero, y_pred_mch_zero),
            'rmse': np.sqrt(mean_squared_error(y_true_mch_zero, y_pred_mch_zero)),
            'mae': mean_absolute_error(y_true_mch_zero, y_pred_mch_zero),
            'y_true': y_true_mch_zero.copy(),
            'y_pred': y_pred_mch_zero.copy(),
            'X': self.boundary_mch_zero_X.copy()
        }
        
        physical_max_1 = np.max(y_true_mch_zero)
        nrmse_1 = calculate_boundary_nrmse(y_pred_mch_zero, y_true_mch_zero, physical_max_1)
        
        self.logger.info(f"  R²:    {results['mch_zero_boundary']['r2']:.6f}")
        self.logger.info(f"  RMSE:  {results['mch_zero_boundary']['rmse']:.6f}")
        self.logger.info(f"  NRMSE: {nrmse_1:.6f}")
        
        # ===== Evaluate Dec=0 Boundary =====
        self.logger.info("Evaluating Dec=0 boundary...")
        
        y_pred_dec_zero = trainer.predict(
            self.boundary_dec_zero_X,
            return_original_scale=True
        ).flatten()
        
        y_true_dec_zero = self.boundary_dec_zero_y_true
        
        results['dec_zero_boundary'] = {
            'r2': r2_score(y_true_dec_zero, y_pred_dec_zero),
            'rmse': np.sqrt(mean_squared_error(y_true_dec_zero, y_pred_dec_zero)),
            'mae': mean_absolute_error(y_true_dec_zero, y_pred_dec_zero),
            'y_true': y_true_dec_zero.copy(),
            'y_pred': y_pred_dec_zero.copy(),
            'X': self.boundary_dec_zero_X.copy()
        }
        
        physical_max_2 = np.max(y_true_dec_zero)
        nrmse_2 = calculate_boundary_nrmse(y_pred_dec_zero, y_true_dec_zero, physical_max_2)
        
        self.logger.info(f"  R²:    {results['dec_zero_boundary']['r2']:.6f}")
        self.logger.info(f"  RMSE:  {results['dec_zero_boundary']['rmse']:.6f}")
        self.logger.info(f"  NRMSE: {nrmse_2:.6f}")
        
        # ===== Evaluate HMN=0 Boundary =====
        self.logger.info("Evaluating HMN=0 boundary...")
        
        y_pred_hmn_zero = trainer.predict(
            self.boundary_hmn_zero_X,
            return_original_scale=True
        ).flatten()
        
        y_true_hmn_zero = self.boundary_hmn_zero_y_true
        
        results['hmn_zero_boundary'] = {
            'r2': r2_score(y_true_hmn_zero, y_pred_hmn_zero),
            'rmse': np.sqrt(mean_squared_error(y_true_hmn_zero, y_pred_hmn_zero)),
            'mae': mean_absolute_error(y_true_hmn_zero, y_pred_hmn_zero),
            'y_true': y_true_hmn_zero.copy(),
            'y_pred': y_pred_hmn_zero.copy(),
            'X': self.boundary_hmn_zero_X.copy()
        }
        
        physical_max_3 = np.max(y_true_hmn_zero)
        nrmse_3 = calculate_boundary_nrmse(y_pred_hmn_zero, y_true_hmn_zero, physical_max_3)
        
        self.logger.info(f"  R²:    {results['hmn_zero_boundary']['r2']:.6f}")
        self.logger.info(f"  RMSE:  {results['hmn_zero_boundary']['rmse']:.6f}")
        self.logger.info(f"  NRMSE: {nrmse_3:.6f}")
        
        # ===== Combined Boundary Score =====
        total_error = nrmse_1 + nrmse_2 + nrmse_3
        boundary_score = exponential_decay_score(total_error, self.decay_lambda)
        
        results['combined'] = {
            'nrmse_mch_zero': nrmse_1,
            'nrmse_dec_zero': nrmse_2,
            'nrmse_hmn_zero': nrmse_3,
            'total_error': total_error,
            'boundary_score': boundary_score,
            'physical_max_mch_zero': physical_max_1,
            'physical_max_dec_zero': physical_max_2,
            'physical_max_hmn_zero': physical_max_3,
            'decay_lambda': self.decay_lambda,
            'scoring_method': 'exponential_decay'
        }
        
        self.logger.info("="*70)
        self.logger.info(f"Combined Boundary Score: {boundary_score:.6f}")
        self.logger.info(
            f"  Total error = {nrmse_1:.4f} + {nrmse_2:.4f} + {nrmse_3:.4f} = {total_error:.4f}"
        )
        self.logger.info(
            f"  Score = exp(-{self.decay_lambda} × {total_error:.4f}) = {boundary_score:.6f}"
        )
        self.logger.info("="*70)
        
        return results


# ==============================================================================
# Thermodynamic Smoothness Evaluator
# ==============================================================================

class ViscositySmoothnessEvaluator:
    """
    Viscosity system thermodynamic smoothness evaluator (4D Laplacian).
    
    Evaluates surface smoothness using 4D Laplacian analysis with P99
    quantile-based roughness metric.
    
    Attributes:
        temp_range: Temperature range (min, max).
        pressure_range: Pressure range (min, max).
        mch_range: MCH concentration range (min, max).
        dec_range: Dec concentration range (min, max).
        grid_resolution: 4D grid resolution tuple (n_T, n_P, n_MCH, n_Dec).
        decay_lambda: Smoothness decay coefficient.
        logger: Logger instance for this evaluator.
    
    Examples:
        >>> evaluator = ViscositySmoothnessEvaluator(
        ...     temp_range=(20, 80),
        ...     pressure_range=(1e5, 1e8),
        ...     grid_resolution=(30, 30, 30, 30)
        ... )
        >>> score, details = evaluator.evaluate_smoothness(trainer)
    """
    
    def __init__(
        self, 
        temp_range: Tuple[float, float] = (20, 80),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        mch_range: Tuple[float, float] = (0, 100),
        dec_range: Tuple[float, float] = (0, 100),
        grid_resolution: Tuple[int, int, int, int] = (30, 30, 30, 30),
        smoothness_decay_lambda: float = 15.0,
        log_level: int = logging.INFO
    ):
        """
        Initialize viscosity smoothness evaluator.
        
        Args:
            temp_range: Temperature range (min, max) in Celsius.
            pressure_range: Pressure range (min, max) in Pa.
            mch_range: MCH concentration range (min, max) in %.
            dec_range: Dec concentration range (min, max) in %.
            grid_resolution: 4D grid resolution (n_T, n_P, n_MCH, n_Dec).
            smoothness_decay_lambda: Smoothness decay coefficient.
            log_level: Logging level.
        """
        self.temp_range = temp_range
        self.pressure_range = pressure_range
        self.mch_range = mch_range
        self.dec_range = dec_range
        self.grid_resolution = grid_resolution
        self.decay_lambda = smoothness_decay_lambda
        self.logger = get_logger(self.__class__.__name__, log_level)
        
    def generate_regular_grid(self) -> np.ndarray:
        """
        Generate 4D regular grid for evaluation.
        
        Returns:
            X_grid: Grid points array [N, 4] with columns [T, P, MCH, Dec].
        
        Examples:
            >>> grid = evaluator.generate_regular_grid()
            >>> print(grid.shape)  # (810000, 4) for (30, 30, 30, 30)
        """
        n_T, n_P, n_MCH, n_Dec = self.grid_resolution
        
        # Generate sampling points for each dimension
        T_samples = np.linspace(*self.temp_range, n_T)
        P_samples = np.linspace(*self.pressure_range, n_P)
        MCH_samples = np.linspace(*self.mch_range, n_MCH)
        Dec_samples = np.linspace(*self.dec_range, n_Dec)
        
        # Create 4D grid
        T_grid, P_grid, MCH_grid, Dec_grid = np.meshgrid(
            T_samples, P_samples, MCH_samples, Dec_samples,
            indexing='ij'
        )
        
        # Flatten to input matrix [N, 4]
        X_grid = np.column_stack([
            T_grid.flatten(),
            P_grid.flatten(),
            MCH_grid.flatten(),
            Dec_grid.flatten()
        ])
        
        self.logger.info("Generated 4D regular grid:")
        self.logger.info(f"  Grid shape: {self.grid_resolution}")
        self.logger.info(f"  Total points: {len(X_grid):,}")
        self.logger.debug(f"  T:   [{self.temp_range[0]}, {self.temp_range[1]}] ({n_T} points)")
        self.logger.debug(f"  P:   [{self.pressure_range[0]:.1e}, {self.pressure_range[1]:.1e}] ({n_P} points)")
        self.logger.debug(f"  MCH: [{self.mch_range[0]}, {self.mch_range[1]}] ({n_MCH} points)")
        self.logger.debug(f"  Dec: [{self.dec_range[0]}, {self.dec_range[1]}] ({n_Dec} points)")
        
        return X_grid
    
    def evaluate_smoothness(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate thermodynamic smoothness using 4D Laplacian method.
        
        Computes the 4D Laplacian operator and uses P99 quantile as a
        robust measure of surface roughness.
        
        Args:
            trainer: PARL trainer object with predict() method.
        
        Returns:
            Tuple of (smoothness_score, detailed_results_dict).
        
        Notes:
            4D Laplacian: ∇²η = ∂²η/∂T² + ∂²η/∂P² + ∂²η/∂MCH² + ∂²η/∂Dec²
        
        Examples:
            >>> score, details = evaluator.evaluate_smoothness(trainer)
            >>> print(f"Smoothness score: {score:.6f}")
            >>> print(f"Roughness (η): {details['normalized_roughness']:.6f}")
        """
        self.logger.info("="*70)
        self.logger.info("Thermodynamic Smoothness Evaluation (4D Laplacian + P99 Quantile)")
        self.logger.info("="*70)
        
        # 1. Generate regular grid
        X_grid = self.generate_regular_grid()
        
        # 2. Model prediction
        self.logger.info("Model prediction in progress...")
        Visc_pred = trainer.predict(X_grid, return_original_scale=True).flatten()
        
        # 3. Reshape back to 4D shape
        Visc_4d = Visc_pred.reshape(self.grid_resolution)
        
        self.logger.info(f"  Prediction complete, 4D tensor shape: {Visc_4d.shape}")
        
        # 4. Calculate 4D Laplacian
        # ∇²η = ∂²η/∂T² + ∂²η/∂P² + ∂²η/∂MCH² + ∂²η/∂Dec²
        
        self.logger.info("Computing 4D Laplacian...")
        
        # Calculate second-order partial derivatives along each dimension
        grad_T = np.gradient(Visc_4d, axis=0)
        d2_dT2 = np.gradient(grad_T, axis=0)
        
        grad_P = np.gradient(Visc_4d, axis=1)
        d2_dP2 = np.gradient(grad_P, axis=1)
        
        grad_MCH = np.gradient(Visc_4d, axis=2)
        d2_dMCH2 = np.gradient(grad_MCH, axis=2)
        
        grad_Dec = np.gradient(Visc_4d, axis=3)
        d2_dDec2 = np.gradient(grad_Dec, axis=3)
        
        # 4D Laplacian
        laplacian_4d = d2_dT2 + d2_dP2 + d2_dMCH2 + d2_dDec2
        laplacian_abs = np.abs(laplacian_4d)
        
        self.logger.debug(f"  Laplacian computed, shape: {laplacian_4d.shape}")
        
        # 5. Compute statistics
        l1_norm = np.mean(laplacian_abs)
        l2_norm = np.sqrt(np.mean(laplacian_4d**2))
        l4_norm = np.power(np.mean(laplacian_abs**4), 0.25)
        linf_norm = np.max(laplacian_abs)
        
        laplacian_p50 = np.percentile(laplacian_abs, 50)
        laplacian_p90 = np.percentile(laplacian_abs, 90)
        laplacian_p95 = np.percentile(laplacian_abs, 95)
        laplacian_p99 = np.percentile(laplacian_abs, 99)
        
        self.logger.debug(f"  L1 norm: {l1_norm:.6f}")
        self.logger.debug(f"  L2 norm: {l2_norm:.6f}")
        self.logger.debug(f"  P99 quantile: {laplacian_p99:.6f}")
        
        # 6. Normalize by data range
        data_range = np.ptp(Visc_4d) + 1e-8
        
        eta_l1 = l1_norm / data_range
        eta_l2 = l2_norm / data_range
        eta_l4 = l4_norm / data_range
        eta_linf = linf_norm / data_range
        
        eta_p50 = laplacian_p50 / data_range
        eta_p90 = laplacian_p90 / data_range
        eta_p95 = laplacian_p95 / data_range
        eta_p99 = laplacian_p99 / data_range
        
        # Use P99 as primary metric
        eta_combined = eta_p99
        
        # 7. Exponential decay scoring
        smoothness_score = np.exp(-self.decay_lambda * eta_combined)
        
        score_if_l2 = np.exp(-self.decay_lambda * eta_l2)
        score_if_l4 = np.exp(-self.decay_lambda * eta_l4)
        score_if_p95 = np.exp(-self.decay_lambda * eta_p95)
        
        self.logger.info(f"  Smoothness score: {smoothness_score:.6f}")
        self.logger.info(f"  Normalized roughness (η_p99): {eta_combined:.6f}")
        
        # 8. Additional diagnostics
        grad_magnitude = np.sqrt(grad_T**2 + grad_P**2 + grad_MCH**2 + grad_Dec**2)
        
        threshold_mild = data_range * 0.02
        threshold_severe = data_range * 0.05
        
        bad_pixels_mild = np.sum(laplacian_abs > threshold_mild) / laplacian_4d.size
        bad_pixels_severe = np.sum(laplacian_abs > threshold_severe) / laplacian_4d.size
        
        # Theoretical validation
        quantile_order_check = (
            laplacian_p50 <= laplacian_p90 <= laplacian_p95 <= 
            laplacian_p99 <= linf_norm
        )
        
        tail_thickness = (laplacian_p99 - laplacian_p95) / (laplacian_p95 + 1e-8)
        
        holder_check_lower = (l2_norm <= l4_norm + 1e-6)
        holder_check_upper = (l4_norm <= linf_norm + 1e-6)
        holder_satisfied = holder_check_lower and holder_check_upper
        
        # 9. Quality assessment
        if smoothness_score >= 0.99:
            quality = 'Excellent'
            description = f'Thermodynamic surface highly smooth (η={eta_combined:.6f})'
        elif smoothness_score >= 0.95:
            quality = 'Good'
            description = f'Surface smooth with minor fluctuations (η={eta_combined:.6f})'
        elif smoothness_score >= 0.90:
            quality = 'Fair'
            description = f'Acceptable smoothness with visible noise (η={eta_combined:.6f})'
        elif smoothness_score >= 0.80:
            quality = 'Poor'
            description = f'Significant roughness detected (η={eta_combined:.6f})'
        else:
            quality = 'Critical'
            description = f'Severe thermodynamic inconsistency (η={eta_combined:.6f})'
        
        # 10. Detailed results
        details = {
            'normalized_roughness': float(eta_p99),
            'smoothness_score': float(smoothness_score),
            
            'laplacian_p50': float(laplacian_p50),
            'laplacian_p90': float(laplacian_p90),
            'laplacian_p95': float(laplacian_p95),
            'laplacian_p99': float(laplacian_p99),
            'laplacian_max': float(linf_norm),
            
            'normalized_p50': float(eta_p50),
            'normalized_p90': float(eta_p90),
            'normalized_p95': float(eta_p95),
            'normalized_p99': float(eta_p99),
            'normalized_max': float(eta_linf),
            
            'l1_norm': float(l1_norm),
            'l2_norm': float(l2_norm),
            'l4_norm': float(l4_norm),
            'linf_norm': float(linf_norm),
            
            'normalized_l1': float(eta_l1),
            'normalized_l2': float(eta_l2),
            'normalized_l4': float(eta_l4),
            'normalized_linf': float(eta_linf),
            
            'score_p99_method': float(smoothness_score),
            'score_l2_method': float(score_if_l2),
            'score_l4_method': float(score_if_l4),
            'score_p95_method': float(score_if_p95),
            
            'score_difference': float(abs(smoothness_score - score_if_l2)),
            'sensitivity_p99_vs_l2': float(eta_p99 / (eta_l2 + 1e-10)),
            'sensitivity_p99_vs_l4': float(eta_p99 / (eta_l4 + 1e-10)),
            
            'tail_thickness': float(tail_thickness),
            'tail_interpretation': (
                'Heavy tail (severe outliers)' if tail_thickness > 0.5 else
                'Moderate tail' if tail_thickness > 0.2 else
                'Light tail (well-behaved)'
            ),
            'p99_to_max_ratio': float(laplacian_p99 / (linf_norm + 1e-10)),
            
            'data_range': float(data_range),
            'data_min': float(np.min(Visc_4d)),
            'data_max': float(np.max(Visc_4d)),
            'data_mean': float(np.mean(Visc_4d)),
            'data_std': float(np.std(Visc_4d)),
            
            'laplacian_mean': float(np.mean(laplacian_4d)),
            'laplacian_std': float(np.std(laplacian_4d)),
            'laplacian_abs_mean': float(np.mean(laplacian_abs)),
            'laplacian_abs_std': float(np.std(laplacian_abs)),
            'laplacian_positive_ratio': float(np.sum(laplacian_4d > 0) / laplacian_4d.size),
            'laplacian_skewness': float(
                np.mean(laplacian_4d**3) / (np.std(laplacian_4d)**3 + 1e-10)
            ),
            'laplacian_kurtosis': float(
                np.mean(laplacian_4d**4) / (np.std(laplacian_4d)**4 + 1e-10)
            ),
            
            'bad_pixels_ratio_mild': float(bad_pixels_mild),
            'bad_pixels_ratio_severe': float(bad_pixels_severe),
            'threshold_mild': float(threshold_mild),
            'threshold_severe': float(threshold_severe),
            
            'gradient_magnitude_mean': float(np.mean(grad_magnitude)),
            'gradient_magnitude_max': float(np.max(grad_magnitude)),
            'gradient_T_rms': float(np.sqrt(np.mean(grad_T**2))),
            'gradient_P_rms': float(np.sqrt(np.mean(grad_P**2))),
            'gradient_MCH_rms': float(np.sqrt(np.mean(grad_MCH**2))),
            'gradient_Dec_rms': float(np.sqrt(np.mean(grad_Dec**2))),
            
            'quantile_order_satisfied': bool(quantile_order_check),
            'holder_inequality_satisfied': bool(holder_satisfied),
            'holder_check_lower': bool(holder_check_lower),
            'holder_check_upper': bool(holder_check_upper),
            'theory_consistency': 'Pass' if (quantile_order_check and holder_satisfied) else 'Warning',
            
            'lambda': self.decay_lambda,
            'grid_resolution': list(self.grid_resolution),
            'actual_shape': list(Visc_4d.shape),
            'total_elements': int(Visc_4d.size),
            
            'coordinate_system': '4D index space',
            'boundary_treatment': 'Full domain (no edge exclusion)',
            'gradient_step': 1.0,
            'method': 'P99 quantile-based (CVaR proxy)',
            'version': '2.0',
            'theory': 'Cahn-Hilliard "almost everywhere" + Quantile Risk Measures',
            'mathematical_foundation': 'P99 as robust proxy for ||∇²η||_{L^∞}',
            'key_advantage': 'Directly measures worst 1% - aligns with "almost everywhere"',
            
            'quality': quality,
            'description': description
        }
        
        self.logger.info("="*70)
        self.logger.info(f"Smoothness Evaluation Complete")
        self.logger.info(f"  Score: {smoothness_score:.6f}")
        self.logger.info(f"  Quality: {quality}")
        self.logger.info("="*70)
        
        return float(smoothness_score), details


# ==============================================================================
# Comprehensive Physics Evaluator
# ==============================================================================

class ViscosityPhysicsEvaluator:
    """
    Comprehensive physics evaluator for viscosity systems (dual-pillar framework).
    
    Combines boundary consistency and thermodynamic smoothness evaluations
    into a unified physics score.
    
    Dual-Pillar Framework:
        1. Boundary consistency: Adherence to binary system limits (3 edges)
        2. Thermodynamic smoothness: Surface quality via 4D Laplacian
    
    Attributes:
        boundary_evaluator: Boundary consistency evaluator instance.
        smoothness_evaluator: Thermodynamic smoothness evaluator instance.
        logger: Logger instance for this evaluator.
    
    Examples:
        >>> evaluator = ViscosityPhysicsEvaluator(
        ...     teacher_models=(model1, model2, model3),
        ...     temp_range=(20, 80),
        ...     pressure_range=(1e5, 1e8)
        ... )
        >>> score, results = evaluator.evaluate_full(trainer)
    """
    
    def __init__(
        self,
        teacher_models: Tuple[Any, Any, Any],
        temp_range: Tuple[float, float] = (20.0, 80.0),
        pressure_range: Tuple[float, float] = (1e5, 1e8),
        mch_range: Tuple[float, float] = (0, 100),
        dec_range: Tuple[float, float] = (0, 100),
        boundary_decay_lambda: float = 5.0,
        smoothness_decay_lambda: float = 15.0,
        n_boundary_samples: int = 100,
        grid_resolution: Tuple[int, int, int, int] = (30, 30, 30, 30),
        log_level: int = logging.INFO
    ):
        """
        Initialize comprehensive physics evaluator.
        
        Args:
            teacher_models: Tuple of (teacher_mch_hmn, teacher_dec_hmn, teacher_mch_dec).
            temp_range: Temperature range (min, max).
            pressure_range: Pressure range (min, max).
            mch_range: MCH concentration range (min, max).
            dec_range: Dec concentration range (min, max).
            boundary_decay_lambda: Decay coefficient for boundary scoring.
            smoothness_decay_lambda: Decay coefficient for smoothness scoring.
            n_boundary_samples: Number of sampling points per boundary.
            grid_resolution: 4D grid resolution for smoothness evaluation.
            log_level: Logging level.
        """
        self.logger = get_logger(self.__class__.__name__, log_level)
        
        teacher_mch_hmn, teacher_dec_hmn, teacher_mch_dec = teacher_models
        
        # Initialize boundary evaluator
        self.boundary_evaluator = ViscosityBoundaryEvaluator(
            teacher_mch_hmn=teacher_mch_hmn,
            teacher_dec_hmn=teacher_dec_hmn,
            teacher_mch_dec=teacher_mch_dec,
            temp_range=temp_range,
            pressure_range=pressure_range,
            decay_lambda=boundary_decay_lambda,
            n_samples=n_boundary_samples,
            log_level=log_level
        )
        
        # Initialize smoothness evaluator
        self.smoothness_evaluator = ViscositySmoothnessEvaluator(
            temp_range=temp_range,
            pressure_range=pressure_range,
            mch_range=mch_range,
            dec_range=dec_range,
            grid_resolution=grid_resolution,
            smoothness_decay_lambda=smoothness_decay_lambda,
            log_level=log_level
        )
        
        self.logger.info("ViscosityPhysicsEvaluator initialized (dual-pillar framework)")
    
    def evaluate_full(self, trainer: Any) -> Tuple[float, Dict[str, Any]]:
        """
        Complete physics evaluation (dual-pillar framework).
        
        Evaluates both boundary consistency and thermodynamic smoothness,
        then combines them into an overall physics score.
        
        Args:
            trainer: PARL trainer object with predict() method.
        
        Returns:
            Tuple of (overall_score, detailed_results_dict) where:
                - overall_score: Geometric mean of boundary and smoothness scores
                - detailed_results_dict: Contains all evaluation details
        
        Examples:
            >>> score, results = evaluator.evaluate_full(trainer)
            >>> print(f"Overall physics score: {score:.6f}")
            >>> print(f"Boundary score: {results['boundary_score']:.6f}")
            >>> print(f"Smoothness score: {results['smoothness_score']:.6f}")
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPREHENSIVE PHYSICS EVALUATION (DUAL-PILLAR FRAMEWORK)")
        self.logger.info("="*70)
        
        # Pillar 1: Boundary consistency
        self.logger.info("\n[PILLAR 1] Boundary Consistency Evaluation")
        boundary_results = self.boundary_evaluator.evaluate_parl_boundary(trainer)
        boundary_score = boundary_results['combined']['boundary_score']
        
        # Pillar 2: Thermodynamic smoothness
        self.logger.info("\n[PILLAR 2] Thermodynamic Smoothness Evaluation")
        smoothness_score, smoothness_details = self.smoothness_evaluator.evaluate_smoothness(trainer)
        
        # Combined score 
        overall_score = 0.5 * boundary_score + 0.5 * smoothness_score
        
        self.logger.info("\n" + "="*70)
        self.logger.info("FINAL PHYSICS EVALUATION RESULTS")
        self.logger.info("="*70)
        self.logger.info(f"  Boundary Score:    {boundary_score:.6f}")
        self.logger.info(f"  Smoothness Score:  {smoothness_score:.6f}")
        self.logger.info(f"  Overall Score:     {overall_score:.6f} (geometric mean)")
        self.logger.info("="*70)
        
        results = {
            'boundary': boundary_results,
            'smoothness': smoothness_details,
            'overall_score': float(overall_score),
            'boundary_score': float(boundary_score),
            'smoothness_score': float(smoothness_score),
            'framework': 'dual-pillar',
            'combination_method': 'geometric_mean'
        }
        
        return overall_score, results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text report from evaluation results.
        
        Args:
            results: Results dictionary from evaluate_full().
        
        Returns:
            Formatted text report string.
        
        Examples:
            >>> score, results = evaluator.evaluate_full(trainer)
            >>> report = evaluator.generate_evaluation_report(results)
            >>> print(report)
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("Viscosity System Physics Evaluation Report")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Overall score
        report_lines.append(f"Overall Physics Score: {results['overall_score']:.6f}")
        report_lines.append("")
        
        # Boundary consistency
        report_lines.append("--- Pillar 1: Boundary Consistency ---")
        boundary = results['boundary']['combined']
        report_lines.append(f"  Boundary Score: {boundary['boundary_score']:.6f}")
        report_lines.append(f"  MCH=0 Boundary NRMSE: {boundary['nrmse_mch_zero']:.6f}")
        report_lines.append(f"  Dec=0 Boundary NRMSE: {boundary['nrmse_dec_zero']:.6f}")
        report_lines.append(f"  HMN=0 Boundary NRMSE: {boundary['nrmse_hmn_zero']:.6f}")
        report_lines.append(f"  Total Error: {boundary['total_error']:.6f}")
        report_lines.append("")
        
        # Thermodynamic smoothness
        report_lines.append("--- Pillar 2: Thermodynamic Smoothness ---")
        smoothness = results['smoothness']
        report_lines.append(f"  Smoothness Score: {smoothness['smoothness_score']:.6f}")
        report_lines.append(f"  Normalized Roughness (η_p99): {smoothness['normalized_roughness']:.6f}")
        report_lines.append(f"  Quality: {smoothness['quality']}")
        report_lines.append(f"  Description: {smoothness['description']}")
        report_lines.append("")
        
        report_lines.append("="*70)
        
        return "\n".join(report_lines)


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Helper functions
    'calculate_boundary_nrmse',
    'exponential_decay_score',
    'get_logger',
    
    # Evaluators
    'ViscosityBoundaryEvaluator',
    'ViscositySmoothnessEvaluator',
    'ViscosityPhysicsEvaluator',
]

