"""
================================================================================
LDPC: Low-Dimensional Physical Constraints - Solubility Systems
================================================================================

Conforming residual correction with exponential decay for ternary solubility
systems. Enforces thermodynamic consistency at binary boundary conditions.

Core Mechanism:
    - Two-boundary correction (left: w_input=0, right: w_input=saturation)
    - Exponential decay: corrections concentrate near boundaries
    - Interior predictions preserved without interference

Mathematical Formula:
    y_corrected = y_base + delta_L * exp(-k*dist_L) + delta_R * exp(-k*dist_R)

where:
    - delta_L: Left boundary correction magnitude
    - delta_R: Right boundary correction magnitude
    - dist_L: Normalized distance from left boundary
    - dist_R: Normalized distance from right boundary
    - k: Decay rate controlling correction range

Decay Rate Physical Meaning:
    - Larger k: Corrections more concentrated near boundaries
    - Smaller k: Corrections extend further into interior
    - Default k=2.0: Decays to <10% at distance ~2 units from boundary

Physical Interpretation:
    For ternary system [T, w_input, w_output]:
    - Left boundary (w_input=0): Output component saturation
    - Right boundary (w_input=saturation): Input component saturation
    
    Examples:
    - KCl-MgCl2-H2O: input=KCl, output=MgCl2
    - NaCl-KCl-H2O: input=NaCl, output=KCl
    - NaCl-MgCl2-H2O: input=NaCl, output=MgCl2

================================================================================
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


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
# LDPC: Conforming Corrected Model for Solubility Systems
# ==============================================================================

class ConformingCorrectedModel(nn.Module):
    """
    LDPC conforming corrected model with exponential decay.
    
    Applies boundary corrections to enforce thermodynamic consistency at
    binary system boundaries using exponential decay strategy. Designed for
    general ternary solubility systems [T, w_input, w_output].
    
    The correction formula:
        y_corrected = y_base + delta_L * exp(-k*dist_L) + delta_R * exp(-k*dist_R)
    
    Key Features:
        - Generic design: works for any ternary system
        - Exponential decay: corrections concentrate near boundaries
        - Linear interpolation mode available for comparison
        - Efficient caching of boundary correction parameters
    
    Attributes:
        base_model: Trained neural network model.
        input_boundary_model: Teacher model for input component saturation boundary.
            Input: T → Output: w_input at saturation (when w_output=0)
        output_boundary_model: Teacher model for output component saturation boundary.
            Input: T → Output: w_output at saturation (when w_input=0)
        x_scaler: Input feature scaler.
        y_scaler: Output feature scaler.
        device: Computing device.
        correction_decay: 'exponential' or 'linear'.
        decay_rate: Exponential decay rate (default: 2.0).
        correction_cache: Cache for boundary corrections.
    
    Examples:
        >>> # Example 1: KCl-MgCl2-H2O system
        >>> # Input: [T, w_KCl], Output: w_MgCl2
        >>> base_model = train_dnn(X_train, y_train)
        >>> corrected_model = ConformingCorrectedModel(
        ...     base_model=base_model,
        ...     input_boundary_model=teacher_kcl,    # KCl saturation when MgCl2=0
        ...     output_boundary_model=teacher_mgcl2, # MgCl2 saturation when KCl=0
        ...     x_scaler=scaler_x,
        ...     y_scaler=scaler_y,
        ...     device='cuda'
        ... )
        >>> 
        >>> # Example 2: NaCl-KCl-H2O system
        >>> # Input: [T, w_NaCl], Output: w_KCl
        >>> corrected_model = ConformingCorrectedModel(
        ...     base_model=base_model,
        ...     input_boundary_model=teacher_nacl,   # NaCl saturation when KCl=0
        ...     output_boundary_model=teacher_kcl,   # KCl saturation when NaCl=0
        ...     x_scaler=scaler_x,
        ...     y_scaler=scaler_y,
        ...     device='cuda'
        ... )
        >>> 
        >>> # Inference with LDPC correction
        >>> corrected_model.eval()
        >>> with torch.no_grad():
        ...     y_pred = corrected_model(x_test)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        input_boundary_model: Callable,
        output_boundary_model: Callable,
        x_scaler: Any,
        y_scaler: Any,
        device: str = 'cuda',
        correction_decay: str = 'exponential',
        decay_rate: float = 2.0
    ) -> None:
        """
        Initialize ConformingCorrectedModel.
        
        Args:
            base_model: Baseline DNN model for ternary system.
            input_boundary_model: Binary boundary teacher model for input component.
                Signature: f(T) → w_input_saturation
                Physical meaning: Saturation curve when output component = 0
                Examples:
                  - KCl-MgCl2 system: f(T) → w_KCl_sat (when w_MgCl2=0)
                  - NaCl-KCl system: f(T) → w_NaCl_sat (when w_KCl=0)
            output_boundary_model: Binary boundary teacher model for output component.
                Signature: f(T) → w_output_saturation
                Physical meaning: Saturation curve when input component = 0
                Examples:
                  - KCl-MgCl2 system: f(T) → w_MgCl2_sat (when w_KCl=0)
                  - NaCl-KCl system: f(T) → w_KCl_sat (when w_NaCl=0)
            x_scaler: Input standardization scaler (e.g., StandardScaler).
            y_scaler: Output standardization scaler (e.g., StandardScaler).
            device: Computing device ('cuda' or 'cpu').
            correction_decay: Correction decay type:
                - 'exponential': Exponential decay (recommended, default)
                - 'linear': Linear interpolation (legacy behavior)
            decay_rate: Exponential decay rate k (only effective in exponential mode).
                - Default 2.0: Decays to <10% at distance 2-3 units from boundary
                - Recommended range: [1.0, 3.0]
                - Physical meaning: Inverse of boundary layer characteristic length
        
        Raises:
            ValueError: If correction_decay is invalid.
            ValueError: If decay_rate is non-positive.
        """
        super().__init__()
        
        self.base_model = base_model
        self.input_boundary_model = input_boundary_model
        self.output_boundary_model = output_boundary_model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device
        
        # Correction strategy parameters
        self.correction_decay = correction_decay
        self.decay_rate = decay_rate
        
        # Correction parameter cache: {temperature → (delta_L, delta_R, w_L, w_R)}
        self.correction_cache: Dict[float, Tuple[float, float, float, float]] = {}
        
        # Parameter validation
        if self.correction_decay not in ['exponential', 'linear']:
            raise ValueError(
                f"correction_decay must be 'exponential' or 'linear', "
                f"got '{correction_decay}'"
            )
        
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
        
        logger.debug(
            f"Initialized ConformingCorrectedModel: "
            f"decay={correction_decay}, rate={decay_rate}"
        )
    
    def _get_correction_params(
        self,
        T_scaled: float
    ) -> Tuple[float, float, float, float]:
        """
        Get correction parameters for given temperature (with caching).
        
        Args:
            T_scaled: Normalized temperature value.
        
        Returns:
            Tuple of (delta_L, delta_R, w_L, w_R):
                - delta_L: Left boundary correction magnitude
                - delta_R: Right boundary correction magnitude
                - w_L: Normalized w_input at left boundary (typically 0)
                - w_R: Normalized w_input at right boundary (saturation)
        
        Notes:
            - Left boundary: w_input=0, predicts output component saturation
            - Right boundary: w_input=saturation, predicts w_output=0
        """
        # Cache key (round to 4 decimals)
        T_key = round(float(T_scaled), 4)
        
        # Check cache
        if T_key in self.correction_cache:
            return self.correction_cache[T_key]
        
        # ========== Transform to physical space ==========
        T_phys = T_scaled * self.x_scaler.scale_[0] + self.x_scaler.mean_[0]
        
        # ========== Left boundary (w_input = 0) ==========
        # Physical meaning: Output component reaches saturation
        w_L_phys = 0.0
        X_L = np.array([[T_phys, w_L_phys]])
        X_L_scaled = self.x_scaler.transform(X_L)
        
        # Base model prediction at left boundary
        with torch.no_grad():
            pred_L = self.base_model(
                torch.FloatTensor(X_L_scaled).to(self.device)
            )
            pred_L = pred_L.cpu().numpy()[0, 0]
        
        # Left boundary target: Output component saturation
        target_L_phys = self.output_boundary_model(np.array([[T_phys]]))[0, 0]
        target_L = self.y_scaler.transform([[target_L_phys]])[0, 0]
        
        # Left boundary correction
        delta_L = target_L - pred_L
        
        # ========== Right boundary (w_input = saturation, w_output = 0) ==========
        # Physical meaning: Input component reaches saturation
        w_R_phys = self.input_boundary_model(np.array([[T_phys]]))[0, 0]
        X_R = np.array([[T_phys, w_R_phys]])
        X_R_scaled = self.x_scaler.transform(X_R)
        
        # Base model prediction at right boundary
        with torch.no_grad():
            pred_R = self.base_model(
                torch.FloatTensor(X_R_scaled).to(self.device)
            )
            pred_R = pred_R.cpu().numpy()[0, 0]
        
        # Right boundary target: w_output = 0
        target_R = self.y_scaler.transform([[0.0]])[0, 0]
        
        # Right boundary correction
        delta_R = target_R - pred_R
        
        # ========== Cache and return ==========
        params = (delta_L, delta_R, X_L_scaled[0, 1], X_R_scaled[0, 1])
        self.correction_cache[T_key] = params
        
        return params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying LDPC conforming correction.
        
        Args:
            x: Input tensor [N, 2] with normalized (T, w_input).
        
        Returns:
            Corrected output [N, 1].
        
        Correction Strategy Comparison:
            Exponential Decay (default):
                correction = delta_L * exp(-k*dist_L) + delta_R * exp(-k*dist_R)
                
                Benefits:
                - Corrections concentrate near boundaries
                - Interior predictions preserved
                - Physically motivated (boundary layer theory)
                
                Example: At w_input=2, exp(-2*2)=0.018, only 1.8% influence
            
            Linear Interpolation (legacy):
                correction = delta_L * (1-u) + delta_R * u
                where u = (w - w_L) / (w_R - w_L)
                
                Issue: At w_input=20, u=0.4, still 60% influence from delta_L
        """
        # ========== Step 1: Base model prediction ==========
        y_base = self.base_model(x)
        
        # ========== Step 2: Apply correction by temperature group ==========
        unique_temps = torch.unique(x[:, 0])
        y_corrected = torch.zeros_like(y_base)
        
        for T_val in unique_temps:
            # Get all samples at this temperature
            mask = (x[:, 0] == T_val)
            
            # Get boundary correction parameters for this temperature
            delta_L, delta_R, w_L, w_R = self._get_correction_params(T_val.item())
            
            # Extract w_input values (normalized space)
            w_batch = x[mask, 1]
            
            # ========== Step 3: Compute correction (strategy-dependent) ==========
            if self.correction_decay == 'exponential':
                # Exponential decay strategy (recommended)
                
                # Distance from left boundary (w_input=0)
                dist_L = torch.abs(w_batch - w_L)
                
                # Distance from right boundary (w_input=saturation)
                dist_R = torch.abs(w_R - w_batch)
                
                # Left boundary weight: exp(-k * dist_L)
                alpha_L = torch.exp(-self.decay_rate * dist_L)
                
                # Right boundary weight: exp(-k * dist_R)
                alpha_R = torch.exp(-self.decay_rate * dist_R)
                
                # Bilateral exponential decay superposition
                # At w_input=0: alpha_L=1, alpha_R≈0 → mainly delta_L
                # At w_input=sat: alpha_L≈0, alpha_R=1 → mainly delta_R
                # In interior: both decay to 0 → minimal correction
                correction = delta_L * alpha_L + delta_R * alpha_R
                
            else:  # 'linear' - maintain original linear interpolation
                # Linear interpolation strategy (legacy)
                u = (w_batch - w_L) / (w_R - w_L + 1e-10)
                u = torch.clamp(u, 0, 1)
                correction = delta_L * (1 - u) + delta_R * u
            
            # ========== Step 4: Apply correction ==========
            y_corrected[mask] = y_base[mask] + correction.unsqueeze(1)
        
        return y_corrected
    
    def clear_cache(self) -> None:
        """
        Clear correction parameter cache.
        
        Use when:
            - Switching teacher models
            - Memory optimization needed
        """
        n_cached = len(self.correction_cache)
        self.correction_cache.clear()
        logger.debug(f"Cleared correction cache ({n_cached} entries)")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get correction cache statistics.
        
        Returns:
            Dictionary containing cache information:
                - cached_temperatures: Number of cached temperature points
                - cache_size_bytes: Approximate cache size in bytes
                - correction_decay: Current correction strategy
                - decay_rate: Current decay rate (if exponential mode)
        """
        return {
            'cached_temperatures': len(self.correction_cache),
            'cache_size_bytes': len(str(self.correction_cache)),
            'correction_decay': self.correction_decay,
            'decay_rate': self.decay_rate if self.correction_decay == 'exponential' else None
        }


# ==============================================================================
# Convenience Creation Function
# ==============================================================================

def create_conforming_corrected_model(
    base_model: nn.Module,
    input_boundary_model: Callable,
    output_boundary_model: Callable,
    x_scaler: Any,
    y_scaler: Any,
    device: str = 'cuda',
    correction_decay: str = 'exponential',
    decay_rate: float = 2.0
) -> ConformingCorrectedModel:
    """
    Create LDPC conforming corrected model for solubility system.
    
    Convenience function for creating ConformingCorrectedModel with
    common settings.
    
    Args:
        base_model: Trained baseline DNN model.
        input_boundary_model: Binary boundary model for input component.
        output_boundary_model: Binary boundary model for output component.
        x_scaler: Input feature scaler.
        y_scaler: Output feature scaler.
        device: Computing device.
        correction_decay: 'exponential' (default) or 'linear'.
        decay_rate: Decay rate for exponential mode (default: 2.0).
    
    Returns:
        Configured ConformingCorrectedModel instance.
    
    Examples:
        >>> # Example 1: KCl-MgCl2-H2O system with exponential decay
        >>> corrected_model = create_conforming_corrected_model(
        ...     base_model, teacher_kcl, teacher_mgcl2,
        ...     x_scaler, y_scaler, device='cuda'
        ... )
        >>> 
        >>> # Example 2: NaCl-KCl-H2O system with custom decay rate
        >>> corrected_model = create_conforming_corrected_model(
        ...     base_model, teacher_nacl, teacher_kcl,
        ...     x_scaler, y_scaler, device='cuda',
        ...     decay_rate=3.0  # Faster decay
        ... )
        >>> 
        >>> # Example 3: Any system with linear interpolation (legacy)
        >>> corrected_model = create_conforming_corrected_model(
        ...     base_model, input_teacher, output_teacher,
        ...     x_scaler, y_scaler, device='cuda',
        ...     correction_decay='linear'
        ... )
    """
    return ConformingCorrectedModel(
        base_model=base_model,
        input_boundary_model=input_boundary_model,
        output_boundary_model=output_boundary_model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        correction_decay=correction_decay,
        decay_rate=decay_rate
    )


# ==============================================================================
# Built-in Testing
# ==============================================================================

def test_conforming_corrected_model():
    """Test ConformingCorrectedModel with dummy data."""
    logger.info("=" * 70)
    logger.info("LDPC Solubility - Testing".center(70))
    logger.info("=" * 70)
    
    # Test 1: Module initialization
    logger.info("\n[Test 1] Module Initialization")
    logger.info("-" * 70)
    
    # Create mock components
    class MockDNN(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1)
    
    class MockScaler:
        def __init__(self):
            self.scale_ = np.array([1.0, 1.0])
            self.mean_ = np.array([0.0, 0.0])
        def transform(self, X):
            return X * self.scale_ + self.mean_
    
    def mock_teacher(X):
        return np.ones((X.shape[0], 1))
    
    base_model = MockDNN()
    scaler = MockScaler()
    
    corrected_model = ConformingCorrectedModel(
        base_model=base_model,
        input_boundary_model=mock_teacher,
        output_boundary_model=mock_teacher,
        x_scaler=scaler,
        y_scaler=scaler,
        decay_rate=2.0
    )
    
    logger.info("Successfully created ConformingCorrectedModel")
    logger.info(f"  Correction decay: {corrected_model.correction_decay}")
    logger.info(f"  Decay rate: {corrected_model.decay_rate}")
    logger.info(f"  Generic parameter naming (input/output boundaries)")
    
    # Test 2: Forward pass
    logger.info("\n[Test 2] Forward Pass")
    logger.info("-" * 70)
    
    x_test = torch.randn(10, 2)
    y_out = corrected_model(x_test)
    
    logger.info(f"Forward pass successful")
    logger.info(f"  Input shape: {x_test.shape}")
    logger.info(f"  Output shape: {y_out.shape}")
    
    # Test 3: Cache functionality
    logger.info("\n[Test 3] Cache Functionality")
    logger.info("-" * 70)
    
    cache_info = corrected_model.get_cache_info()
    logger.info(f"Cache info retrieved")
    logger.info(f"  Cached temperatures: {cache_info['cached_temperatures']}")
    
    corrected_model.clear_cache()
    cache_info_after = corrected_model.get_cache_info()
    logger.info(f"Cache cleared")
    logger.info(f"  Cached temperatures: {cache_info_after['cached_temperatures']}")
    
    # Test 4: Correction modes
    logger.info("\n[Test 4] Correction Modes")
    logger.info("-" * 70)
    
    # Exponential mode
    model_exp = ConformingCorrectedModel(
        base_model, mock_teacher, mock_teacher,
        scaler, scaler, correction_decay='exponential'
    )
    logger.info("Exponential mode: OK")
    
    # Linear mode
    model_lin = ConformingCorrectedModel(
        base_model, mock_teacher, mock_teacher,
        scaler, scaler, correction_decay='linear'
    )
    logger.info("Linear mode: OK")
    
    # Test 5: Generic parameter naming test
    logger.info("\n[Test 5] Generic Parameter Naming")
    logger.info("-" * 70)
    
    # Simulate different ternary systems
    systems = [
        ("KCl-MgCl2-H2O", "teacher_kcl", "teacher_mgcl2"),
        ("NaCl-KCl-H2O", "teacher_nacl", "teacher_kcl"),
        ("NaCl-MgCl2-H2O", "teacher_nacl", "teacher_mgcl2")
    ]
    
    for sys_name, input_name, output_name in systems:
        model = ConformingCorrectedModel(
            base_model, mock_teacher, mock_teacher,
            scaler, scaler
        )
        logger.info(f"{sys_name} system can use same interface")
        logger.info(f"  input_boundary_model={input_name}")
        logger.info(f"  output_boundary_model={output_name}")
    

if __name__ == "__main__":
    test_conforming_corrected_model()