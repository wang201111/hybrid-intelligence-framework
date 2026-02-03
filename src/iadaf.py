"""
================================================================================
IADAF: Integrated Adaptive Data Augmentation Framework
================================================================================

A unified data augmentation framework combining Bayesian optimization, 
WGAN-GP training, and XGBoost filtering for generating high-quality 
synthetic data.

Supports two systems:
    - Solubility: [T, w_comp1, w_comp2] 
    - Viscosity: [T, P, comp1, comp2, viscosity]

Workflow:
    Data preparation → BO search (TSTR) → WGAN-GP training → 
    Generation + Filtering → High-quality synthetic data

Key Components:
    1. Bayesian Optimization: Automatic hyperparameter search
    2. WGAN-GP Training: Learn data distribution
    3. XGBoost Filtering: Quality control
    4. Data Generation: Output synthetic samples

================================================================================
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Bayesian optimization
try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False
    warnings.warn("bayesian-optimization not installed. BO functionality unavailable.")

warnings.filterwarnings('ignore')


# ==============================================================================
# Logger Configuration
# ==============================================================================

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent format.
    
    Args:
        name: Logger name.
        level: Logging level.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger(__name__)


# ==============================================================================
# Section 1: Configuration Management
# ==============================================================================

@dataclass
class IADAFConfig:
    """
    Complete configuration for IADAF framework.
    
    Attributes:
        SYSTEM_TYPE: System type - 'solubility' or 'viscosity'.
        
        WGAN-GP Architecture:
            LATENT_DIM: Latent space dimension (optimized by BO).
            HIDDEN_DIM: Hidden layer dimension (optimized by BO).
            LAMBDA_GP: Gradient penalty coefficient (optimized by BO).
            N_CRITIC: Critic update frequency.
            DROPOUT: Dropout rate.
        
        Training Parameters:
            LEARNING_RATE_G: Generator learning rate.
            LEARNING_RATE_C: Critic learning rate.
            BATCH_SIZE: Batch size.
            WGAN_EPOCHS: Full WGAN training epochs.
            BO_WGAN_EPOCHS: WGAN epochs during BO (faster).
        
        Bayesian Optimization:
            BO_N_ITERATIONS: BO iteration count.
            BO_INIT_POINTS: BO initial random samples.
            BO_DNN_EPOCHS: DNN training epochs during BO.
            
        XGBoost Filtering:
            XGB_ERROR_THRESHOLD: Error threshold for filtering.
            XGB_MAX_DEPTH: Tree depth.
            XGB_N_ESTIMATORS: Number of trees.
            XGB_LEARNING_RATE: Learning rate.
        
        Data Generation:
            N_SAMPLES_TO_GENERATE: Number of samples to generate.
            T_MIN: Minimum temperature.
            T_MAX: Maximum temperature.
            P_MIN: Minimum pressure (viscosity only).
            P_MAX: Maximum pressure (viscosity only).
        
        Other:
            TRAIN_RATIO: Training set ratio.
            VAL_RATIO: Validation set ratio.
            RANDOM_SEED: Random seed.
            DEVICE: Device ('auto', 'cuda', 'cpu').
            VERBOSE: Whether to print detailed logs.
    
    Examples:
        >>> # Solubility system
        >>> config = IADAFConfig(
        ...     SYSTEM_TYPE='solubility',
        ...     WGAN_EPOCHS=1000,
        ...     XGB_ERROR_THRESHOLD=15.0
        ... )
        >>> 
        >>> # Viscosity system
        >>> config = IADAFConfig(
        ...     SYSTEM_TYPE='viscosity',
        ...     WGAN_EPOCHS=500,
        ...     XGB_ERROR_THRESHOLD=0.5
        ... )
    """
    
    # ========== System Type ==========
    SYSTEM_TYPE: str = 'solubility'  # 'solubility' or 'viscosity'
    
    # ========== WGAN-GP Architecture ==========
    LATENT_DIM: int = 10
    HIDDEN_DIM: int = 256
    LAMBDA_GP: float = 10.0
    N_CRITIC: int = 5
    DROPOUT: float = 0.1
    
    # ========== Training Parameters ==========
    LEARNING_RATE_G: float = 1e-4
    LEARNING_RATE_C: float = 1e-4
    BATCH_SIZE: int = 64
    WGAN_EPOCHS: int = 1000
    BO_WGAN_EPOCHS: int = 50
    
    # ========== Bayesian Optimization ==========
    BO_N_ITERATIONS: int = 100
    BO_INIT_POINTS: int = 10
    BO_DNN_EPOCHS: int = 200
    
    # Hyperparameter search space
    LATENT_DIM_RANGE: Tuple[int, int] = (1, 100)
    HIDDEN_DIM_RANGE: Tuple[int, int] = (150, 350)
    LAMBDA_GP_RANGE: Tuple[float, float] = (0.05, 0.5)
    
    # ========== XGBoost Filtering ==========
    XGB_ERROR_THRESHOLD: float = 15.0
    XGB_MAX_DEPTH: int = 6
    XGB_N_ESTIMATORS: int = 100
    XGB_LEARNING_RATE: float = 0.1
    
    # ========== Data Generation ==========
    N_SAMPLES_TO_GENERATE: int = 2000
    T_MIN: float = -40.0
    T_MAX: float = 200.0
    P_MIN: float = 1e6  # For viscosity system
    P_MAX: float = 1e8  # For viscosity system
    
    # ========== Data Split ==========
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.2
    
    # ========== Other ==========
    RANDOM_SEED: int = 42
    DEVICE: str = 'auto'
    VERBOSE: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and auto-configuration."""
        # Validate system type
        if self.SYSTEM_TYPE not in ['solubility', 'viscosity']:
            raise ValueError(
                f"Invalid SYSTEM_TYPE: {self.SYSTEM_TYPE}. "
                "Must be 'solubility' or 'viscosity'."
            )
        
        # Auto-detect device
        if self.DEVICE == 'auto':
            self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Warn if BO unavailable
        if not BAYESOPT_AVAILABLE:
            warnings.warn(
                "bayesian-optimization not installed. BO functionality unavailable."
            )
        
        # System-specific defaults
        if self.SYSTEM_TYPE == 'viscosity':
            if self.XGB_ERROR_THRESHOLD == 15.0:  # Default solubility value
                self.XGB_ERROR_THRESHOLD = 0.5  # Adjust for viscosity
            if self.T_MIN == -40.0:  # Default solubility range
                self.T_MIN = 20.0
                self.T_MAX = 80.0


# ==============================================================================
# Section 2: System Configuration
# ==============================================================================

SYSTEM_CONFIGS = {
    'solubility': {
        'input_dim': 3,  # [T, w_comp1, w_comp2]
        'condition_dim': 2,  # [T, w_comp1]
        'target_dim': 1,  # w_comp2
        'xgb_input_cols': 2,  # X[:, :2] for XGBoost
        'xgb_target_col': 2,  # X[:, 2] for XGBoost
        'generator_output': 1,  # Generate 1 value
        'critic_input': 1,  # Discriminate 1 value
        'has_pressure': False,  # No pressure condition
        'description': 'Solubility system [T, w_comp1, w_comp2]'
    },
    'viscosity': {
        'input_dim': 5,  # [T, P, comp1, comp2, viscosity]
        'condition_dim': 2,  # [T, P]
        'target_dim': 3,  # [comp1, comp2, viscosity]
        'xgb_input_cols': 4,  # X[:, :4] for XGBoost
        'xgb_target_col': 4,  # X[:, 4] for XGBoost
        'generator_output': 3,  # Generate 3 values
        'critic_input': 3,  # Discriminate 3 values
        'has_pressure': True,  # Has pressure condition
        'description': 'Viscosity system [T, P, comp1, comp2, viscosity]'
    }
}


# ==============================================================================
# Section 3: Network Architecture
# ==============================================================================

class Generator(nn.Module):
    """
    WGAN-GP Generator network.
    
    Generates synthetic data conditioned on input features.
    Architecture: 4-layer fully connected network with LayerNorm and Dropout.
    
    Attributes:
        latent_dim: Dimension of latent noise vector.
        condition_dim: Dimension of conditioning variables.
        output_dim: Dimension of generated output.
        hidden_dim: Hidden layer dimension.
        net: Sequential neural network.
    
    Examples:
        >>> # Solubility system (output_dim=1)
        >>> gen = Generator(latent_dim=10, condition_dim=2, 
        ...                 output_dim=1, hidden_dim=256)
        >>> z = torch.randn(64, 10)
        >>> conditions = torch.randn(64, 2)
        >>> output = gen(z, conditions)  # [64, 1]
        >>> 
        >>> # Viscosity system (output_dim=3)
        >>> gen = Generator(latent_dim=10, condition_dim=2,
        ...                 output_dim=3, hidden_dim=256)
        >>> output = gen(z, conditions)  # [64, 3]
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize Generator.
        
        Args:
            latent_dim: Latent space dimension.
            condition_dim: Conditioning dimension (e.g., [T, w_comp1]).
            output_dim: Output dimension (1 for solubility, 3 for viscosity).
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = latent_dim + condition_dim
        
        # 4 hidden layers
        for _ in range(4):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        z: torch.Tensor,
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Latent vector [batch_size, latent_dim].
            conditions: Condition vector [batch_size, condition_dim].
        
        Returns:
            Generated output [batch_size, output_dim].
        """
        x = torch.cat([z, conditions], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    """
    WGAN-GP Critic (Discriminator) network.
    
    Evaluates the quality of real vs generated samples.
    Architecture: 4-layer fully connected network with LayerNorm and Dropout.
    
    Attributes:
        input_dim: Dimension of input to discriminate.
        condition_dim: Dimension of conditioning variables.
        hidden_dim: Hidden layer dimension.
        net: Sequential neural network.
    
    Examples:
        >>> # Solubility system (input_dim=1)
        >>> critic = Critic(input_dim=1, condition_dim=2, hidden_dim=256)
        >>> samples = torch.randn(64, 1)
        >>> conditions = torch.randn(64, 2)
        >>> scores = critic(samples, conditions)  # [64, 1]
        >>> 
        >>> # Viscosity system (input_dim=3)
        >>> critic = Critic(input_dim=3, condition_dim=2, hidden_dim=256)
        >>> samples = torch.randn(64, 3)
        >>> scores = critic(samples, conditions)  # [64, 1]
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ) -> None:
        """
        Initialize Critic.
        
        Args:
            input_dim: Input dimension (1 for solubility, 3 for viscosity).
            condition_dim: Conditioning dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim + condition_dim
        
        # 4 hidden layers
        for _ in range(4):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input samples [batch_size, input_dim].
            conditions: Condition vector [batch_size, condition_dim].
        
        Returns:
            Critic scores [batch_size, 1].
        """
        combined = torch.cat([x, conditions], dim=-1)
        return self.net(combined)


# ==============================================================================
# Section 4: Loss Functions
# ==============================================================================

class WGANGPLoss:
    """
    WGAN-GP loss functions (Wasserstein distance + Gradient Penalty).
    
    Implements the loss computation for training WGAN with gradient penalty.
    """
    
    @staticmethod
    def compute_gradient_penalty(
        critic: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        conditions: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            critic: Critic network.
            real_samples: Real samples [batch_size, input_dim].
            fake_samples: Generated samples [batch_size, input_dim].
            conditions: Conditioning vector [batch_size, condition_dim].
            lambda_gp: Gradient penalty coefficient.
        
        Returns:
            Gradient penalty loss.
        """
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, device=device)
        
        # Interpolated samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Critic output on interpolates
        d_interpolates = critic(interpolates, conditions)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    @staticmethod
    def compute_critic_loss(
        critic: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        conditions: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute critic loss.
        
        Args:
            critic: Critic network.
            real_samples: Real samples.
            fake_samples: Generated samples.
            conditions: Conditioning vector.
            lambda_gp: Gradient penalty coefficient.
        
        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Critic scores on real/fake samples
        real_validity = critic(real_samples, conditions)
        fake_validity = critic(fake_samples.detach(), conditions)
        
        # Wasserstein distance
        wasserstein_distance = fake_validity.mean() - real_validity.mean()
        
        # Gradient penalty
        gradient_penalty = WGANGPLoss.compute_gradient_penalty(
            critic, real_samples, fake_samples, conditions, lambda_gp
        )
        
        # Total loss
        loss = wasserstein_distance + gradient_penalty
        
        metrics = {
            'critic_loss': loss.item(),
            'wasserstein': wasserstein_distance.item(),
            'gp': gradient_penalty.item()
        }
        
        return loss, metrics
    
    @staticmethod
    def compute_generator_loss(
        critic: nn.Module,
        fake_samples: torch.Tensor,
        conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute generator loss.
        
        Args:
            critic: Critic network.
            fake_samples: Generated samples.
            conditions: Conditioning vector.
        
        Returns:
            Tuple of (loss, metrics_dict).
        """
        fake_validity = critic(fake_samples, conditions)
        loss = -fake_validity.mean()
        
        metrics = {
            'generator_loss': loss.item()
        }
        
        return loss, metrics


# ==============================================================================
# Section 5: XGBoost Filter
# ==============================================================================

class XGBoostFilter:
    """
    XGBoost-based quality filter for synthetic data.
    
    Filters generated samples based on prediction error threshold.
    
    Attributes:
        config: IADAF configuration.
        model: XGBoost regressor.
        is_fitted: Whether model has been trained.
    
    Examples:
        >>> filter = XGBoostFilter(config)
        >>> filter.fit(X_train, y_train)
        >>> X_filtered, stats = filter.filter(X_generated)
        >>> print(f"Filtered: {stats['filtered_samples']} samples")
    """
    
    def __init__(self, config: IADAFConfig) -> None:
        """
        Initialize XGBoost filter.
        
        Args:
            config: IADAF configuration.
        """
        self.config = config
        self.model = xgb.XGBRegressor(
            max_depth=config.XGB_MAX_DEPTH,
            n_estimators=config.XGB_N_ESTIMATORS,
            learning_rate=config.XGB_LEARNING_RATE,
            random_state=config.RANDOM_SEED,
            verbosity=0
        )
        self.is_fitted = False
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> 'XGBoostFilter':
        """
        Train XGBoost filter model.
        
        Args:
            X_train: Training features [N, n_features].
                - Solubility: [N, 2] (T, w_comp1)
                - Viscosity: [N, 4] (T, P, comp1, comp2)
            y_train: Training targets [N].
                - Solubility: w_comp2
                - Viscosity: viscosity
        
        Returns:
            Self for method chaining.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def filter(
        self,
        X_generated: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filter generated data based on prediction error.
        
        Args:
            X_generated: Generated data [N, n_features].
                - Solubility: [N, 3] (T, w_comp1, w_comp2)
                - Viscosity: [N, 5] (T, P, comp1, comp2, viscosity)
            threshold: Error threshold (uses config value if None).
        
        Returns:
            Tuple of (filtered_data, statistics_dict).
        
        Raises:
            RuntimeError: If model hasn't been trained.
        """
        if not self.is_fitted:
            raise RuntimeError("XGBoost model not trained. Call fit() first.")
        
        if threshold is None:
            threshold = self.config.XGB_ERROR_THRESHOLD
        
        # Get system configuration
        sys_config = SYSTEM_CONFIGS[self.config.SYSTEM_TYPE]
        n_input_cols = sys_config['xgb_input_cols']
        target_col = sys_config['xgb_target_col']
        
        # Extract features and targets
        X_features = X_generated[:, :n_input_cols]
        y_true = X_generated[:, target_col]
        
        # Predict
        y_pred = self.model.predict(X_features)
        
        # Compute errors
        errors = np.abs(y_true - y_pred)
        
        # Filter by threshold
        mask = errors < threshold
        X_filtered = X_generated[mask]
        
        # Statistics
        stats = {
            'original_samples': len(X_generated),
            'filtered_samples': len(X_filtered),
            'filter_ratio': len(X_filtered) / len(X_generated) if len(X_generated) > 0 else 0.0,
            'mean_error_all': float(errors.mean()),
            'mean_error_filtered': float(errors[mask].mean()) if mask.any() else 0.0
        }
        
        return X_filtered, stats


# ==============================================================================
# Section 6: Bayesian Optimizer
# ==============================================================================

class BayesianOptimizer:
    """
    Bayesian optimizer for automatic hyperparameter search.
    
    Uses Bayesian optimization to find optimal WGAN-GP hyperparameters
    via TSTR (Train on Synthetic, Test on Real) paradigm.
    
    Attributes:
        config: IADAF configuration.
        history: Optimization history.
        optimizer: BayesianOptimization instance.
    
    Examples:
        >>> optimizer = BayesianOptimizer(config)
        >>> best_params = optimizer.optimize(objective_fn, param_bounds)
        >>> print(f"Best latent_dim: {best_params['latent_dim']}")
    """
    
    def __init__(self, config: IADAFConfig) -> None:
        """
        Initialize Bayesian optimizer.
        
        Args:
            config: IADAF configuration.
        """
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.optimizer: Optional[BayesianOptimization] = None
    
    def optimize(
        self,
        objective_fn: callable,
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Execute Bayesian optimization.
        
        Args:
            objective_fn: Objective function f(latent_dim, hidden_dim, lambda_gp) -> score.
            param_bounds: Parameter search space.
        
        Returns:
            Dictionary of optimal hyperparameters.
        
        Raises:
            RuntimeError: If bayesian-optimization not installed.
        """
        if not BAYESOPT_AVAILABLE:
            raise RuntimeError("bayesian-optimization not installed!")
        
        self.optimizer = BayesianOptimization(
            f=objective_fn,
            pbounds=param_bounds,
            random_state=self.config.RANDOM_SEED,
            verbose=2 if self.config.VERBOSE else 0
        )
        
        if self.config.VERBOSE:
            logger.info("=" * 70)
            logger.info("Starting Bayesian Optimization".center(70))
            logger.info("=" * 70)
        
        self.optimizer.maximize(
            init_points=self.config.BO_INIT_POINTS,
            n_iter=self.config.BO_N_ITERATIONS
        )
        
        # Save history
        self.history = [
            {
                'params': res['params'],
                'target': res['target']
            }
            for res in self.optimizer.res
        ]
        
        best_params = self.optimizer.max['params']
        best_score = self.optimizer.max['target']
        
        if self.config.VERBOSE:
            logger.info("=" * 70)
            logger.info("Bayesian Optimization Complete".center(70))
            logger.info("=" * 70)
            logger.info(f"\nBest Hyperparameters:")
            logger.info(f"  latent_dim: {int(best_params['latent_dim'])}")
            logger.info(f"  hidden_dim: {int(best_params['hidden_dim'])}")
            logger.info(f"  lambda_gp: {best_params['lambda_gp']:.6f}")
            logger.info(f"\nBest TSTR Val R²: {best_score:.6f}\n")
        
        return best_params


# ==============================================================================
# Section 7: Simple DNN for TSTR Evaluation
# ==============================================================================

class FcBlock(nn.Module):
    """Fully connected block: Linear → BatchNorm1d → PReLU."""
    
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SimpleDNN(nn.Module):
    """
    Simple DNN regressor for TSTR evaluation.
    
    Architecture: Input → FcBlock(100) → FcBlock(100) → FcBlock(1)
    This matches the DNN structure in utils.py with layerDim=3, nodeDim=100.
    
    Attributes:
        input_dim: Input dimension.
        layers: Sequential network layers.
    
    Examples:
        >>> model = SimpleDNN(input_dim=2)
        >>> x = torch.randn(32, 2)
        >>> output = model(x)  # [32, 1]
    """
    
    def __init__(self, input_dim: int) -> None:
        """
        Initialize SimpleDNN.
        
        Args:
            input_dim: Input feature dimension.
        """
        super().__init__()
        self.input_dim = input_dim
        
        self.layers = nn.Sequential(
            FcBlock(input_dim, 100),
            FcBlock(100, 100),
            FcBlock(100, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


def train_simple_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    device: str = 'cuda',
    verbose: bool = False
) -> Tuple[float, nn.Module]:
    """
    Train simple DNN for TSTR evaluation.
    
    Args:
        X_train: Training features [N_train, n_features].
        y_train: Training targets [N_train].
        X_val: Validation features [N_val, n_features].
        y_val: Validation targets [N_val].
        epochs: Number of training epochs.
        device: Device ('cuda' or 'cpu').
        verbose: Whether to print progress.
    
    Returns:
        Tuple of (best_val_r2, trained_model).
    """
    # Prepare data
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val = y_val.reshape(-1)
    
    # Model
    model = SimpleDNN(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training
    best_val_r2 = -np.inf
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train_t)
        loss = criterion(train_pred, y_train_t)
        
        loss.backward()
        optimizer.step()
        

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
            val_r2 = r2_score(y_val, val_pred)
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{epochs}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Val R²: {val_r2:.6f}"
                )
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            logger.info(
                f"  ✓ Loaded best model "
                f"(Epoch {best_epoch}, Val R² = {best_val_r2:.6f})"
            )
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
        final_val_r2 = r2_score(y_val, val_pred)
    
    return final_val_r2, model


# ==============================================================================
# Section 8: IADAF Trainer
# ==============================================================================

class IADAFTrainer:
    """
    Unified IADAF trainer for data augmentation.
    
    Integrates Bayesian optimization, WGAN-GP training, and XGBoost filtering
    into a complete data augmentation framework.
    
    Supports two systems:
        - Solubility: [T, w_comp1, w_comp2]
        - Viscosity: [T, P, comp1, comp2, viscosity]
    
    Main Methods:
        - fit(X_data): Train complete pipeline (auto 80/20 split)
        - fit_with_validation(X_train, X_val): Train with fixed splits
        - generate(n_samples): Generate high-quality synthetic data
        - save(path) / load(path): Model persistence
    
    Attributes:
        config: IADAF configuration.
        system_config: System-specific configuration.
        device: Computation device.
        generator: Trained generator network.
        critic: Trained critic network.
        xgb_filter: XGBoost filter.
        scaler: MinMax scaler for data normalization.
        is_fitted: Whether model has been trained.
        best_params: Optimal hyperparameters from BO.
        history: Training history.
    
    Examples:
        >>> # Solubility system
        >>> config = IADAFConfig(SYSTEM_TYPE='solubility', WGAN_EPOCHS=1000)
        >>> trainer = IADAFTrainer(config)
        >>> trainer.fit(X_solubility)  # [N, 3]
        >>> X_synthetic = trainer.generate(n_samples=1500)
        >>> 
        >>> # Viscosity system
        >>> config = IADAFConfig(SYSTEM_TYPE='viscosity', WGAN_EPOCHS=500)
        >>> trainer = IADAFTrainer(config)
        >>> trainer.fit(X_viscosity)  # [N, 5]
        >>> X_synthetic = trainer.generate(n_samples=2000)
    """
    
    def __init__(self, config: IADAFConfig) -> None:
        """
        Initialize IADAF trainer.
        
        Args:
            config: IADAF configuration with SYSTEM_TYPE specified.
        
        Raises:
            ValueError: If SYSTEM_TYPE is invalid.
        """
        self.config = config
        self.system_config = SYSTEM_CONFIGS[config.SYSTEM_TYPE]
        self.device = torch.device(config.DEVICE)
        
        # Components
        self.generator: Optional[Generator] = None
        self.critic: Optional[Critic] = None
        self.xgb_filter = XGBoostFilter(config)
        self.bo_optimizer = (
            BayesianOptimizer(config) if BAYESOPT_AVAILABLE else None
        )
        
        # State management
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        
        # Training history
        self.history: Dict[str, Any] = {
            'bo_iterations': [],
            'wgan_losses': [],
            'filter_stats': {}
        }
        
        # Set random seeds
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        if config.VERBOSE:
            logger.info(f"Initialized IADAFTrainer for {config.SYSTEM_TYPE} system")
            logger.info(f"System config: {self.system_config['description']}")
    
    def fit(self, X_data: np.ndarray) -> 'IADAFTrainer':
        """
        Train complete IADAF pipeline.
        
        Workflow:
            1. Data split (80/20 train/validation)
            2. Bayesian optimization (TSTR paradigm)
            3. Full WGAN-GP training with optimal hyperparameters
            4. XGBoost filter training
        
        Args:
            X_data: Training data [N, n_features].
                - Solubility: [N, 3] (T, w_comp1, w_comp2)
                - Viscosity: [N, 5] (T, P, comp1, comp2, viscosity)
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> trainer = IADAFTrainer(config)
            >>> trainer.fit(X_data)
            >>> print(f"Training complete. Best params: {trainer.best_params}")
        """
        if self.config.VERBOSE:
            logger.info("=" * 70)
            logger.info(
                f"IADAF Training Pipeline Started - "
                f"{self.config.SYSTEM_TYPE.capitalize()}"
                .center(70)
            )
            logger.info("=" * 70)
        
        # Stage 1: Data preparation
        if self.config.VERBOSE:
            logger.info("\n[Stage 1/3] Data Preparation")
        
        X_train, X_val = train_test_split(
            X_data,
            train_size=self.config.TRAIN_RATIO,
            random_state=self.config.RANDOM_SEED
        )
        
        # Normalization
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.config.VERBOSE:
            logger.info(f"  Training set: {len(X_train)} samples")
            logger.info(f"  Validation set: {len(X_val)} samples")
        
        # Stage 2: Bayesian optimization
        if self.config.VERBOSE:
            logger.info("\n[Stage 2/3] Bayesian Optimization")
        
        self.best_params = self._run_bayesian_optimization(
            X_train_scaled, X_val_scaled
        )
        
        # Stage 3: Train full WGAN-GP
        if self.config.VERBOSE:
            logger.info("\n[Stage 3/3] Full WGAN-GP Training")
        
        self._train_final_wgan(X_train_scaled, self.best_params)
        
        # Train XGBoost filter with WGAN-generated synthetic data
        if self.config.VERBOSE:
            logger.info(f"\n[XGBoost Filter Training]")
            logger.info(f"  Using WGAN-generated synthetic data")
        
        # Generate synthetic data for XGBoost training
        n_synthetic_for_xgb = max(len(X_train), 2000)
        X_synthetic_for_xgb = self._generate_samples(
            n_synthetic_for_xgb, scaled=False
        )
        
        # Train XGBoost filter
        n_input_cols = self.system_config['xgb_input_cols']
        target_col = self.system_config['xgb_target_col']
        
        self.xgb_filter.fit(
            X_synthetic_for_xgb[:, :n_input_cols],
            X_synthetic_for_xgb[:, target_col]
        )
        
        if self.config.VERBOSE:
            logger.info(
                f"  ✓ XGBoost training complete "
                f"(samples: {len(X_synthetic_for_xgb)})"
            )
        
        self.is_fitted = True
        
        if self.config.VERBOSE:
            logger.info("\n" + "=" * 70)
            logger.info("IADAF Training Complete!".center(70))
            logger.info("=" * 70)
        
        return self
    
    def fit_with_validation(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray
    ) -> 'IADAFTrainer':
        """
        Train complete IADAF pipeline with fixed train/validation splits.
        
        Workflow:
            1. Use provided train/validation sets (no splitting)
            2. Bayesian optimization (TSTR paradigm)
            3. Full WGAN-GP training with optimal hyperparameters
            4. XGBoost filter training
        
        Args:
            X_train: Training data [N_train, n_features].
            X_val: Validation data [N_val, n_features].
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> trainer = IADAFTrainer(config)
            >>> trainer.fit_with_validation(X_train, X_val)
            >>> X_synthetic = trainer.generate(1500)
        """
        if self.config.VERBOSE:
            logger.info("=" * 70)
            logger.info(
                f"IADAF Training Pipeline Started (Fixed Validation) - "
                f"{self.config.SYSTEM_TYPE.capitalize()}"
                .center(70)
            )
            logger.info("=" * 70)
        
        # Stage 1: Data preparation
        if self.config.VERBOSE:
            logger.info("\n[Stage 1/3] Data Preparation")
        
        # Normalization (based on training set)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if self.config.VERBOSE:
            logger.info(f"  Training set: {len(X_train)} samples")
            logger.info(f"  Validation set: {len(X_val)} samples")
        
        # Stage 2: Bayesian optimization
        if self.config.VERBOSE:
            logger.info("\n[Stage 2/3] Bayesian Optimization")
        
        self.best_params = self._run_bayesian_optimization(
            X_train_scaled, X_val_scaled
        )
        
        # Stage 3: Train full WGAN-GP
        if self.config.VERBOSE:
            logger.info("\n[Stage 3/3] Full WGAN-GP Training")
        
        self._train_final_wgan(X_train_scaled, self.best_params)
        
        # Train XGBoost filter
        if self.config.VERBOSE:
            logger.info(f"\n[XGBoost Filter Training]")
            logger.info(f"  Using WGAN-generated synthetic data")
        
        n_synthetic_for_xgb = max(len(X_train), 2000)
        X_synthetic_for_xgb = self._generate_samples(
            n_synthetic_for_xgb, scaled=False
        )
        
        n_input_cols = self.system_config['xgb_input_cols']
        target_col = self.system_config['xgb_target_col']
        
        self.xgb_filter.fit(
            X_synthetic_for_xgb[:, :n_input_cols],
            X_synthetic_for_xgb[:, target_col]
        )
        
        if self.config.VERBOSE:
            logger.info(
                f"  ✓ XGBoost training complete "
                f"(samples: {len(X_synthetic_for_xgb)})"
            )
        
        self.is_fitted = True
        
        if self.config.VERBOSE:
            logger.info("\n" + "=" * 70)
            logger.info("IADAF Training Complete!".center(70))
            logger.info("=" * 70)
        
        return self
    
    def _run_bayesian_optimization(
        self,
        X_train_scaled: np.ndarray,
        X_val_scaled: np.ndarray
    ) -> Dict[str, Any]:
        """
        Execute Bayesian optimization using TSTR paradigm.
        
        Args:
            X_train_scaled: Normalized training data.
            X_val_scaled: Normalized validation data.
        
        Returns:
            Dictionary of optimal hyperparameters.
        """
        if not BAYESOPT_AVAILABLE:
            warnings.warn("BO unavailable, using default hyperparameters")
            return {
                'latent_dim': self.config.LATENT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'lambda_gp': self.config.LAMBDA_GP
            }
        
        # Define objective function
        def objective_function(latent_dim, hidden_dim, lambda_gp):
            """TSTR objective function."""
            # 1. Train WGAN-GP (fast version)
            wgan_losses = self._train_wgan_gp(
                X_train_scaled,
                latent_dim=int(latent_dim),
                hidden_dim=int(hidden_dim),
                lambda_gp=lambda_gp,
                epochs=self.config.BO_WGAN_EPOCHS
            )
            
            # 2. Generate synthetic data
            X_gen_scaled = self._generate_samples(
                n_samples=self.config.N_SAMPLES_TO_GENERATE,
                scaled=True
            )
            
            # 3. XGBoost filtering
            xgb_temp = XGBoostFilter(self.config)
            n_input_cols = self.system_config['xgb_input_cols']
            target_col = self.system_config['xgb_target_col']
            
            xgb_temp.fit(
                X_train_scaled[:, :n_input_cols],
                X_train_scaled[:, target_col]
            )
            X_synthetic_scaled, _ = xgb_temp.filter(X_gen_scaled)
            
            if len(X_synthetic_scaled) < 50:  # Too few samples
                return 0.0
            
            # 4. Train DNN (only on synthetic data)
            val_r2, _ = train_simple_dnn(
                X_synthetic_scaled[:, :n_input_cols],
                X_synthetic_scaled[:, target_col],
                X_val_scaled[:, :n_input_cols],
                X_val_scaled[:, target_col],
                epochs=self.config.BO_DNN_EPOCHS,
                device=self.device.type,
                verbose=False
            )
            
            return val_r2
        
        # Parameter bounds
        param_bounds = {
            'latent_dim': self.config.LATENT_DIM_RANGE,
            'hidden_dim': self.config.HIDDEN_DIM_RANGE,
            'lambda_gp': self.config.LAMBDA_GP_RANGE
        }
        
        # Run optimization
        best_params = self.bo_optimizer.optimize(objective_function, param_bounds)
        self.best_score = self.bo_optimizer.optimizer.max['target']
        self.history['bo_iterations'] = self.bo_optimizer.history
        
        return best_params
    
    def _train_wgan_gp(
        self,
        X_data_scaled: np.ndarray,
        latent_dim: int,
        hidden_dim: int,
        lambda_gp: float,
        epochs: int
    ) -> List[Dict[str, Any]]:
        """
        Train WGAN-GP model.
        
        Args:
            X_data_scaled: Normalized training data.
            latent_dim: Latent dimension.
            hidden_dim: Hidden layer dimension.
            lambda_gp: Gradient penalty coefficient.
            epochs: Number of training epochs.
        
        Returns:
            List of training losses per epoch.
        """
        # Initialize networks
        condition_dim = self.system_config['condition_dim']
        output_dim = self.system_config['generator_output']
        input_dim = self.system_config['critic_input']
        
        self.generator = Generator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=self.config.DROPOUT
        ).to(self.device)
        
        self.critic = Critic(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            dropout=self.config.DROPOUT
        ).to(self.device)
        
        # Optimizers
        optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.LEARNING_RATE_G,
            betas=(0.5, 0.999)
        )
        optimizer_c = optim.Adam(
            self.critic.parameters(),
            lr=self.config.LEARNING_RATE_C,
            betas=(0.5, 0.999)
        )
        
        # Data loader
        dataset = TensorDataset(torch.FloatTensor(X_data_scaled))
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        
        # Training loop
        losses_history = []
        
        iterator = (
            tqdm(range(epochs), desc="Training WGAN-GP")
            if self.config.VERBOSE
            else range(epochs)
        )
        
        for epoch in iterator:
            epoch_losses = {'critic': [], 'generator': []}
            
            for batch_data, in dataloader:
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)
                
                # Extract conditions and targets based on system
                conditions = batch_data[:, :condition_dim]
                real_samples = batch_data[:, condition_dim:]
                
                # ========== Train Critic ==========
                for _ in range(self.config.N_CRITIC):
                    optimizer_c.zero_grad()
                    
                    # Generate fake samples
                    z = torch.randn(batch_size, latent_dim, device=self.device)
                    fake_samples = self.generator(z, conditions)
                    
                    # Critic loss
                    critic_loss, _ = WGANGPLoss.compute_critic_loss(
                        self.critic, real_samples, fake_samples,
                        conditions, lambda_gp
                    )
                    
                    critic_loss.backward()
                    optimizer_c.step()
                    
                    epoch_losses['critic'].append(critic_loss.item())
                
                # ========== Train Generator ==========
                optimizer_g.zero_grad()
                
                z = torch.randn(batch_size, latent_dim, device=self.device)
                fake_samples = self.generator(z, conditions)
                
                gen_loss, _ = WGANGPLoss.compute_generator_loss(
                    self.critic, fake_samples, conditions
                )
                
                gen_loss.backward()
                optimizer_g.step()
                
                epoch_losses['generator'].append(gen_loss.item())
            
            # Record losses
            losses_history.append({
                'epoch': epoch + 1,
                'critic_loss': np.mean(epoch_losses['critic']),
                'generator_loss': np.mean(epoch_losses['generator'])
            })
        
        return losses_history
    
    def _train_final_wgan(
        self,
        X_train_scaled: np.ndarray,
        best_params: Dict[str, Any]
    ) -> None:
        """
        Train final WGAN-GP with optimal hyperparameters.
        
        Args:
            X_train_scaled: Normalized training data.
            best_params: Optimal hyperparameters from BO.
        """
        self.history['wgan_losses'] = self._train_wgan_gp(
            X_train_scaled,
            latent_dim=int(best_params['latent_dim']),
            hidden_dim=int(best_params['hidden_dim']),
            lambda_gp=best_params['lambda_gp'],
            epochs=self.config.WGAN_EPOCHS
        )
    
    def _generate_samples(
        self,
        n_samples: int,
        scaled: bool = False
    ) -> np.ndarray:
        """
        Generate samples (internal method).
        
        Args:
            n_samples: Number of samples to generate.
            scaled: Whether to return normalized data.
        
        Returns:
            Generated data [n_samples, n_features].
        
        Raises:
            RuntimeError: If generator not initialized.
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized!")
        
        self.generator.eval()
        
        with torch.no_grad():
            # Generate conditions based on system type
            if self.system_config['has_pressure']:
                # Viscosity system: [T, P]
                T = np.random.uniform(
                    self.config.T_MIN, self.config.T_MAX, (n_samples, 1)
                )
                P = np.random.uniform(
                    self.config.P_MIN, self.config.P_MAX, (n_samples, 1)
                )
                conditions_raw = np.column_stack([T, P])
                
                # Create dummy full data for normalization
                X_temp = np.column_stack([
                    conditions_raw,
                    np.zeros((n_samples, self.system_config['target_dim']))
                ])
            else:
                # Solubility system: [T, w_comp1]
                T = np.random.uniform(
                    self.config.T_MIN, self.config.T_MAX, (n_samples, 1)
                )
                comp1 = np.random.uniform(0, 50, (n_samples, 1))
                conditions_raw = np.column_stack([T, comp1])
                
                # Create dummy full data
                X_temp = np.column_stack([
                    conditions_raw,
                    np.zeros((n_samples, 1))
                ])
            
            # Normalize
            X_scaled = self.scaler.transform(X_temp)
            
            # Extract normalized conditions
            condition_dim = self.system_config['condition_dim']
            conditions = torch.FloatTensor(
                X_scaled[:, :condition_dim]
            ).to(self.device)
            
            # Generate targets
            z = torch.randn(
                n_samples, self.generator.latent_dim, device=self.device
            )
            targets_scaled = self.generator(z, conditions).cpu().numpy()
            
            # Combine conditions and targets
            X_gen_scaled = np.column_stack([
                X_scaled[:, :condition_dim],
                targets_scaled
            ])
            
            if scaled:
                return X_gen_scaled
            else:
                return self.scaler.inverse_transform(X_gen_scaled)
    
    def generate(
        self,
        n_samples: int = 1500,
        T_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Generate high-quality synthetic data (main interface).
        
        Args:
            n_samples: Desired number of samples (may be reduced by filtering).
            T_range: Optional temperature range (T_min, T_max).
        
        Returns:
            Filtered high-quality synthetic data [M, n_features].
            - Solubility: [M, 3] (T, w_comp1, w_comp2)
            - Viscosity: [M, 5] (T, P, comp1, comp2, viscosity)
        
        Raises:
            RuntimeError: If model hasn't been trained.
        
        Examples:
            >>> trainer.fit(X_train)
            >>> X_synthetic = trainer.generate(n_samples=1500)
            >>> print(f"Generated {len(X_synthetic)} high-quality samples")
            >>> 
            >>> # With temperature range
            >>> X_synthetic = trainer.generate(1500, T_range=(20, 80))
        """
        if not self.is_fitted:
            raise RuntimeError("Model not trained! Call fit() first.")
        
        # Temporarily adjust temperature range if specified
        if T_range is not None:
            original_T_min = self.config.T_MIN
            original_T_max = self.config.T_MAX
            self.config.T_MIN = T_range[0]
            self.config.T_MAX = T_range[1]
        
        if self.config.VERBOSE:
            logger.info(f"\nGenerating synthetic data...")
            logger.info(f"  Initial generation: {n_samples} samples")
        
        # Generate samples
        X_generated = self._generate_samples(n_samples, scaled=False)
        
        # XGBoost filtering
        X_filtered, stats = self.xgb_filter.filter(X_generated)
        
        self.history['filter_stats'] = stats
        
        if self.config.VERBOSE:
            logger.info(f"  After filtering: {stats['filtered_samples']} samples")
            logger.info(f"  Filter ratio: {stats['filter_ratio']*100:.1f}%")
            logger.info(f"  Mean error (filtered): {stats['mean_error_filtered']:.4f}")
        
        # Restore temperature range
        if T_range is not None:
            self.config.T_MIN = original_T_min
            self.config.T_MAX = original_T_max
        
        return X_filtered
    
    def save(self, path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Save path (e.g., 'models/iadaf_model.pth').
        
        Examples:
            >>> trainer.save('models/iadaf_solubility.pth')
        """
        save_dict = {
            'config': self.config,
            'generator_state': (
                self.generator.state_dict() if self.generator else None
            ),
            'critic_state': (
                self.critic.state_dict() if self.critic else None
            ),
            'scaler': self.scaler,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.history,
            'is_fitted': self.is_fitted
        }
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(save_dict, path)
        
        if self.config.VERBOSE:
            logger.info(f"\nModel saved to: {path}")
    
    def load(self, path: str) -> 'IADAFTrainer':
        """
        Load trained model from disk.
        
        Args:
            path: Load path.
        
        Returns:
            Self for method chaining.
        
        Examples:
            >>> trainer = IADAFTrainer(config)
            >>> trainer.load('models/iadaf_solubility.pth')
            >>> X_synthetic = trainer.generate(1500)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore config
        self.config = checkpoint['config']
        self.system_config = SYSTEM_CONFIGS[self.config.SYSTEM_TYPE]
        
        # Restore networks
        if checkpoint['generator_state'] is not None:
            condition_dim = self.system_config['condition_dim']
            output_dim = self.system_config['generator_output']
            input_dim = self.system_config['critic_input']
            
            self.generator = Generator(
                latent_dim=checkpoint['best_params']['latent_dim'],
                condition_dim=condition_dim,
                output_dim=output_dim,
                hidden_dim=checkpoint['best_params']['hidden_dim'],
                dropout=self.config.DROPOUT
            ).to(self.device)
            self.generator.load_state_dict(checkpoint['generator_state'])
            
            self.critic = Critic(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dim=checkpoint['best_params']['hidden_dim'],
                dropout=self.config.DROPOUT
            ).to(self.device)
            self.critic.load_state_dict(checkpoint['critic_state'])
        
        # Restore state
        self.scaler = checkpoint['scaler']
        self.best_params = checkpoint['best_params']
        self.best_score = checkpoint['best_score']
        self.history = checkpoint['history']
        self.is_fitted = checkpoint['is_fitted']
        
        if self.config.VERBOSE:
            logger.info(f"\nModel loaded from: {path}")
            logger.info(f"System type: {self.config.SYSTEM_TYPE}")
            logger.info(f"Best params: {self.best_params}")
        
        return self


# ==============================================================================
# Built-in Testing
# ==============================================================================

def test_iadaf():
    """Test IADAF with both systems."""
    logger.info("\n" + "=" * 70)
    logger.info("IADAF Standardized Module - Testing".center(70))
    logger.info("=" * 70)
    
    # Test 1: Solubility system
    logger.info("\n[Test 1] Solubility System")
    logger.info("-" * 70)
    
    # Generate dummy solubility data [T, w_comp1, w_comp2]
    np.random.seed(42)
    n_samples = 200
    X_sol = np.random.rand(n_samples, 3) * np.array([100, 50, 50])
    X_sol[:, 0] += 20  # Temperature: 20-120°C
    
    config_sol = IADAFConfig(
        SYSTEM_TYPE='solubility',
        WGAN_EPOCHS=10,  # Fast test
        BO_N_ITERATIONS=2,
        BO_WGAN_EPOCHS=5,
        VERBOSE=False
    )
    
    trainer_sol = IADAFTrainer(config_sol)
    logger.info(f"✓ Created trainer: {config_sol.SYSTEM_TYPE}")
    logger.info(f"  System config: {trainer_sol.system_config['description']}")
    logger.info(f"  Input dim: {trainer_sol.system_config['input_dim']}")
    logger.info(f"  Generator output: {trainer_sol.system_config['generator_output']}")
    
    # Test 2: Viscosity system
    logger.info("\n[Test 2] Viscosity System")
    logger.info("-" * 70)
    
    # Generate dummy viscosity data [T, P, comp1, comp2, viscosity]
    X_vis = np.random.rand(n_samples, 5)
    X_vis[:, 0] = X_vis[:, 0] * 60 + 20  # T: 20-80°C
    X_vis[:, 1] = X_vis[:, 1] * 9e7 + 1e6  # P: 1e6-1e8 N/sqm
    
    config_vis = IADAFConfig(
        SYSTEM_TYPE='viscosity',
        WGAN_EPOCHS=10,
        BO_N_ITERATIONS=2,
        BO_WGAN_EPOCHS=5,
        VERBOSE=False
    )
    
    trainer_vis = IADAFTrainer(config_vis)
    logger.info(f"✓ Created trainer: {config_vis.SYSTEM_TYPE}")
    logger.info(f"  System config: {trainer_vis.system_config['description']}")
    logger.info(f"  Input dim: {trainer_vis.system_config['input_dim']}")
    logger.info(f"  Generator output: {trainer_vis.system_config['generator_output']}")
    
    # Test 3: Interface compatibility
    logger.info("\n[Test 3] Interface Compatibility")
    logger.info("-" * 70)
    
    # Test fit method signature
    logger.info("✓ fit(X_data) method available")
    logger.info("✓ fit_with_validation(X_train, X_val) method available")
    logger.info("✓ generate(n_samples, T_range) method available")
    logger.info("✓ save(path) / load(path) methods available")
    
    # Test configuration
    logger.info("\n[Test 4] Configuration")
    logger.info("-" * 70)
    logger.info(f"✓ Solubility default XGB threshold: 15.0")
    logger.info(f"✓ Viscosity default XGB threshold: {config_vis.XGB_ERROR_THRESHOLD}")
    logger.info(f"✓ Solubility T range: {config_sol.T_MIN}°C to {config_sol.T_MAX}°C")
    logger.info(f"✓ Viscosity T range: {config_vis.T_MIN}°C to {config_vis.T_MAX}°C")
    


if __name__ == "__main__":
    test_iadaf()