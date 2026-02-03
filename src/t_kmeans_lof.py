"""Temperature-based K-Means + LOF Outlier Detection Module.

This module implements a unified framework for cleaning experimental data in
multi-component chemical systems. It combines K-Means clustering on temperature
with Local Outlier Factor (LOF) detection and Robust Moving Median (RMM)
residual analysis.

The module supports two types of systems:
    - Solubility (Ternary): 3 input columns [Temperature, w_comp1, w_comp2]
    - Viscosity (Quaternary): 4 input columns [Temperature, comp1, comp2, comp3]

All detection follows a unified pipeline:
    1. Temperature-based K-Means clustering
    2. Local anomaly detection (LOF + Moving Median Residual)
    3. Global consensus scoring and ranking
    4. Top-N removal based on contamination rate

Standardized for publication in Nature Communications.

Example:
    >>> from tkmeans_lof import TKMeansLOF
    >>> 
    >>> # Solubility system
    >>> cleaner = TKMeansLOF(
    ...     config={'lof_contamination': 0.20},
    ...     system_type='solubility',
    ...     verbose=True
    ... )
    >>> X_clean, y_clean, outliers, info = cleaner.clean(X, y)
    >>> 
    >>> # Viscosity system
    >>> cleaner = TKMeansLOF(
    ...     config={'lof_contamination': 0.05},
    ...     system_type='viscosity',
    ...     verbose=True
    ... )
    >>> X_clean, y_clean, outliers, info = cleaner.clean(X, y)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

# Configure module-level logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class OutlierDetectionConfig:
    """Configuration parameters for TKMeansLOF outlier detector.
    
    Attributes:
        contamination: Expected proportion of outliers in dataset [0.0, 0.5].
            If None, uses system-specific default (solubility: 0.20, viscosity: 0.05).
        n_neighbors: Number of neighbors for LOF algorithm. Range: [5, 30].
        min_cluster_size: Minimum cluster size for processing. Clusters smaller
            than this value are skipped.
        auto_select_k: Whether to automatically select optimal number of clusters
            using silhouette score.
        min_n_clusters: Minimum number of clusters to try.
        max_n_clusters: Maximum number of clusters to try. Will be capped by
            system-specific defaults (solubility: 24, viscosity: 15).
        fixed_n_clusters: Manual cluster count (used when auto_select_k=False).
        random_state: Random seed for reproducibility.
        enable_residual_detection: Whether to enable moving median residual
            detection. Both systems use this by default.
        moving_median_window: Window size for moving median computation.
        consensus_weight: Weight for LOF score in consensus scoring.
            consensus = w * LOF + (1-w) * Residual, where w = consensus_weight.
    """
    
    contamination: Optional[float] = None
    n_neighbors: int = 5
    min_cluster_size: int = 10
    
    auto_select_k: bool = True
    min_n_clusters: int = 2
    max_n_clusters: int = 24
    fixed_n_clusters: Optional[int] = None
    
    random_state: int = 42
    
    enable_residual_detection: bool = True
    moving_median_window: int = 5
    consensus_weight: float = 0.5


# System-specific default configurations
SYSTEM_DEFAULTS = {
    'solubility': {
        'contamination': 0.20,
        'max_n_clusters': 24,
        'n_input_cols': 3,
        'lof_cols_idx': [1, 2],
        'residual_sort_col_idx': 0,  # Sort by comp1 (w_comp1)
        'description': 'Solubility system [Temperature, w_comp1, w_comp2]'
    },
    'viscosity': {
        'contamination': 0.05,
        'max_n_clusters': 15,
        'n_input_cols': 4,
        'lof_cols_idx': [1, 2, 3],
        'residual_sort_col_idx': 1,  # Sort by comp2
        'description': 'Viscosity system [Temperature, comp1, comp2, comp3]'
    }
}


# =============================================================================
# Main Class
# =============================================================================

class TKMeansLOF:
    """Temperature-based K-Means + LOF outlier detector.
    
    This class implements a consensus-based outlier detection framework that
    combines multiple detection strategies:
        1. K-Means clustering on temperature dimension
        2. Local Outlier Factor (LOF) for density-based detection
        3. Robust Moving Median (RMM) residual for trend-deviation detection
        4. Consensus scoring to combine both views
        5. Global ranking and top-N removal
    
    The detector automatically adapts to system type while maintaining a
    unified interface and processing pipeline.
    
    Attributes:
        config: Configuration object containing all parameters.
        system_type: System type ('solubility' or 'viscosity').
        system_defaults: System-specific default parameters.
        outlier_indices_: Indices of detected outliers (set after clean()).
        info_: Dictionary with detection statistics (set after clean()).
    
    Examples:
        Basic usage with solubility data:
        
        >>> import numpy as np
        >>> from tkmeans_lof import TKMeansLOF
        >>> 
        >>> # Prepare data: 3 columns [T, w_comp1, w_comp2]
        >>> X = np.random.rand(500, 3)
        >>> y = np.random.rand(500)
        >>> 
        >>> # Create detector with default settings
        >>> cleaner = TKMeansLOF(
        ...     config={},
        ...     system_type='solubility'
        ... )
        >>> 
        >>> # Clean data
        >>> X_clean, y_clean, outliers, info = cleaner.clean(X, y)
        >>> print(f"Removed {info['n_outliers']} outliers")
        
        Advanced usage with custom configuration:
        
        >>> # Custom configuration
        >>> config = {
        ...     'lof_contamination': 0.15,
        ...     'lof_n_neighbors': 5,
        ...     'auto_select_k': True,
        ...     'min_cluster_size': 10
        ... }
        >>> 
        >>> cleaner = TKMeansLOF(
        ...     config=config,
        ...     system_type='viscosity',
        ...     verbose=True
        ... )
        >>> 
        >>> X_clean, y_clean, outliers, info = cleaner.clean(X, y)
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], OutlierDetectionConfig],
        system_type: str = 'solubility',
        verbose: bool = True,
        target_indices: Optional[List[int]] = None
    ) -> None:
        """Initialize the outlier detector.
        
        Args:
            config: Configuration dictionary or OutlierDetectionConfig object.
                Dictionary keys should use legacy naming (e.g., 'lof_contamination')
                for backward compatibility.
            system_type: Type of chemical system to process.
                Must be 'solubility' or 'viscosity'.
            verbose: If True, enables detailed logging at INFO level.
            target_indices: Legacy parameter for tracking specific data points.
                Kept for backward compatibility but not actively used.
        
        Raises:
            ValueError: If system_type is not 'solubility' or 'viscosity'.
        
        Note:
            When config is a dictionary, legacy parameter names are automatically
            mapped to new names:
                - 'lof_contamination' -> 'contamination'
                - 'lof_n_neighbors' -> 'n_neighbors'
                - 'lof_weight' -> triggers residual detection with 0.5 weight
        """
        # Validate system type
        if system_type not in ['solubility', 'viscosity']:
            raise ValueError(
                f"Invalid system_type: '{system_type}'. "
                "Must be 'solubility' or 'viscosity'."
            )
        
        self.system_type = system_type
        self.system_defaults = SYSTEM_DEFAULTS[system_type]
        self.target_indices = target_indices or []
        
        # Process configuration
        if isinstance(config, dict):
            self.config = self._create_config_from_dict(config)
        else:
            self.config = config
        
        # Apply system-specific defaults
        self._apply_system_defaults()
        
        # Configure logging
        self._configure_logging(verbose)
        
        # Attributes to be populated after clean()
        self.outlier_indices_: Optional[np.ndarray] = None
        self.info_: Dict[str, Any] = {}
        
        logger.info(
            f"Initialized TKMeansLOF for {system_type} system "
            f"(contamination={self.config.contamination:.2f})"
        )
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> OutlierDetectionConfig:
        """Create OutlierDetectionConfig from legacy dictionary.
        
        Args:
            config_dict: Configuration dictionary with legacy parameter names.
        
        Returns:
            OutlierDetectionConfig object with parameters mapped from dictionary.
        
        Note:
            Performs automatic mapping of legacy parameter names:
            - 'lof_contamination' -> 'contamination'
            - 'lof_n_neighbors' -> 'n_neighbors'
        """
        return OutlierDetectionConfig(
            contamination=config_dict.get('lof_contamination', None),
            n_neighbors=config_dict.get('lof_n_neighbors', 5),
            min_cluster_size=config_dict.get('min_cluster_size', 10),
            auto_select_k=config_dict.get('auto_select_k', True),
            min_n_clusters=config_dict.get('min_n_clusters', 2),
            max_n_clusters=config_dict.get('max_n_clusters', 24),
            fixed_n_clusters=config_dict.get('fixed_n_clusters', None),
            random_state=config_dict.get('random_state', 42),
            enable_residual_detection=config_dict.get('enable_residual_detection', True),
            moving_median_window=config_dict.get('moving_median_window', 5),
            consensus_weight=config_dict.get('consensus_weight', 0.5)
        )
    
    def _apply_system_defaults(self) -> None:
        """Apply system-specific default values to configuration.
        
        Sets contamination rate and max_n_clusters based on system type
        if not explicitly specified by user.
        """
        # Apply default contamination if not specified
        if self.config.contamination is None:
            self.config.contamination = self.system_defaults['contamination']
            logger.debug(
                f"Using default contamination for {self.system_type}: "
                f"{self.config.contamination:.2f}"
            )
        
        # Cap max_n_clusters by system default
        system_max = self.system_defaults['max_n_clusters']
        if self.config.max_n_clusters > system_max:
            logger.warning(
                f"max_n_clusters ({self.config.max_n_clusters}) exceeds "
                f"system default ({system_max}). Using {system_max}."
            )
            self.config.max_n_clusters = system_max
    
    def _configure_logging(self, verbose: bool) -> None:
        """Configure logging handler and level.
        
        Args:
            verbose: If True, set logging level to INFO, otherwise WARNING.
        """
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    def clean(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute outlier detection and data cleaning pipeline.
        
        This method implements the complete three-stage pipeline:
            1. Temperature-based K-Means clustering
            2. LOF + Moving Median Residual detection per cluster
            3. Global consensus scoring and top-N removal
        
        Args:
            X: Input feature matrix.
                - Solubility: shape (n_samples, 3) [Temperature, w_comp1, w_comp2]
                - Viscosity: shape (n_samples, 4) [Temperature, comp1, comp2, comp3]
                Note: First column must always be temperature.
            y: Target values array, shape (n_samples,).
        
        Returns:
            Tuple containing:
                - X_clean: Cleaned feature matrix with outliers removed.
                - y_clean: Cleaned target array with outliers removed.
                - outlier_indices: Original indices of detected outliers.
                - info: Dictionary with detection statistics containing:
                    - 'n_outliers': Number of outliers removed
                    - 'outlier_rate': Proportion of outliers (0 to 1)
                    - 'iterations': Number of iterations (always 1)
                    - 'n_clusters': Number of temperature clusters used
                    - 'n_clusters_processed': Clusters with size > min_cluster_size
                    - 'n_clusters_skipped': Clusters with size <= min_cluster_size
                    - 'method': Detection method name
        
        Raises:
            ValueError: If X has wrong number of columns for specified system type.
            ValueError: If X and y have different number of samples.
        
        Examples:
            >>> cleaner = TKMeansLOF(config={}, system_type='solubility')
            >>> X_clean, y_clean, outliers, info = cleaner.clean(X, y)
            >>> print(f"Removed {len(outliers)} outliers ({info['outlier_rate']:.1%})")
        """
        # Validate input dimensions
        self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        
        logger.info("="*70)
        logger.info(f"Starting TKMeansLOF cleaning - {self.system_type} system")
        logger.info(f"Original samples: {n_samples}")
        logger.info(f"Input shape: {X.shape} ({self.system_defaults['description']})")
        logger.info(f"Detection mode: LOF + Moving Median Residual (Consensus)")
        logger.info(
            f"LOF parameters: contamination={self.config.contamination:.2f}, "
            f"n_neighbors={self.config.n_neighbors}"
        )
        logger.info(
            f"Residual parameters: window={self.config.moving_median_window}, "
            f"weight={self.config.consensus_weight:.2f}"
        )
        logger.info(f"Min cluster size: {self.config.min_cluster_size}")
        logger.info("="*70)
        
        # Create DataFrame with standardized column names
        df = self._create_dataframe(X, y)
        
        # Stage 1: Temperature clustering
        self._cluster_by_temperature(df)
        
        # Stage 2: Local anomaly detection (LOF + Residual)
        self._detect_local_anomalies(df)
        
        # Stage 3: Global consensus scoring and filtering
        X_clean, y_clean, outlier_indices = self._filter_outliers(df, X, y)
        
        # Update instance attributes
        self.outlier_indices_ = outlier_indices
        
        # Log final results
        logger.info("="*70)
        logger.info("[Cleaning Results]")
        logger.info("="*70)
        logger.info(f"Original samples: {n_samples}")
        logger.info(
            f"Cleaned samples: {len(X_clean)} "
            f"({len(X_clean)/n_samples*100:.2f}%)"
        )
        logger.info(
            f"Removed outliers: {self.info_['n_outliers']} "
            f"({self.info_['outlier_rate']*100:.2f}%)"
        )
        logger.info(
            f"Processed clusters: {self.info_['n_clusters_processed']}/"
            f"{self.info_['n_clusters']}"
        )
        logger.info(
            f"Skipped clusters: {self.info_['n_clusters_skipped']}/"
            f"{self.info_['n_clusters']}"
        )
        logger.info(f"Method: {self.info_['method']}")
        logger.info(f"Output shape: {X_clean.shape}")
        logger.info("="*70)
        
        return X_clean, y_clean, outlier_indices, self.info_
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input data dimensions.
        
        Args:
            X: Input feature matrix.
            y: Target values array.
        
        Raises:
            ValueError: If X has wrong number of columns for system type.
            ValueError: If X and y have different number of samples.
        """
        n_samples, n_features = X.shape
        expected_cols = self.system_defaults['n_input_cols']
        
        if n_features != expected_cols:
            raise ValueError(
                f"Invalid input shape for {self.system_type} system. "
                f"Expected {expected_cols} columns, got {n_features}. "
                f"Input should be: {self.system_defaults['description']}"
            )
        
        if len(y) != n_samples:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {n_samples}, y: {len(y)}"
            )
    
    def _create_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Create DataFrame with standardized column names.
        
        Args:
            X: Input feature matrix.
            y: Target values array.
        
        Returns:
            DataFrame with columns:
                - 'temperature': Temperature values (column 0 of X)
                - 'comp1', 'comp2', ... : Component values (remaining columns)
                - 'target': Target values (y)
                - 'original_index': Original row indices for tracking
        """
        n_features = X.shape[1]
        
        # Create column names: ['temperature', 'comp1', 'comp2', ...]
        col_names = ['temperature'] + [f'comp{i}' for i in range(1, n_features)]
        
        df = pd.DataFrame(X, columns=col_names)
        df['target'] = y
        df['original_index'] = np.arange(len(df))
        
        return df
    
    def _cluster_by_temperature(self, df: pd.DataFrame) -> None:
        """Perform K-Means clustering on temperature dimension.
        
        Args:
            df: DataFrame with temperature column and other features.
        
        Note:
            Modifies df in-place by adding 'temp_cluster' column.
            Updates self.info_ with cluster count.
        """
        logger.info("="*70)
        logger.info("[Stage 1] Temperature Clustering (K-Means)")
        logger.info("="*70)
        
        X_temp = df[['temperature']].values
        
        if self.config.auto_select_k:
            n_clusters = self._auto_select_clusters(X_temp)
        else:
            n_clusters = (
                self.config.fixed_n_clusters 
                if self.config.fixed_n_clusters 
                else self.config.min_n_clusters
            )
            logger.info(f"Using fixed cluster count: {n_clusters}")
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10
        )
        df['temp_cluster'] = kmeans.fit_predict(X_temp)
        
        self.info_['n_clusters'] = n_clusters
        
        # Log cluster statistics
        logger.info("\nCluster statistics:")
        for cluster_id in sorted(df['temp_cluster'].unique()):
            cluster_data = df[df['temp_cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            temp_range = cluster_data['temperature']
            temp_min, temp_max = temp_range.min(), temp_range.max()
            logger.info(
                f"  Cluster {cluster_id:2d}: {cluster_size:4d} points "
                f"(T: {temp_min:7.2f} ~ {temp_max:7.2f})"
            )
    
    def _auto_select_clusters(self, X_temp: np.ndarray) -> int:
        """Automatically select optimal number of clusters using silhouette score.
        
        Args:
            X_temp: Temperature values, shape (n_samples, 1).
        
        Returns:
            Optimal number of clusters.
        """
        logger.info(
            f"Auto-selecting clusters (range: {self.config.min_n_clusters}-"
            f"{self.config.max_n_clusters})..."
        )
        
        best_score = -1.0
        best_k = self.config.min_n_clusters
        
        # Limit search range to avoid over-segmentation
        max_k = min(
            self.config.max_n_clusters,
            len(X_temp) // self.config.min_cluster_size
        )
        max_k = max(max_k, self.config.min_n_clusters + 1)
        
        silhouette_scores = []
        for k in range(self.config.min_n_clusters, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(X_temp)
            
            # Need at least 2 unique labels for silhouette score
            if len(np.unique(labels)) < 2:
                continue
            
            score = silhouette_score(X_temp, labels)
            silhouette_scores.append((k, score))
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info(
            f"Optimal cluster count: {best_k} "
            f"(silhouette score: {best_score:.4f})"
        )
        
        return best_k
    
    def _detect_local_anomalies(self, df: pd.DataFrame) -> None:
        """Detect local anomalies using LOF and moving median residual.
        
        This method processes each temperature cluster independently:
            1. Apply LOF on composition features
            2. Compute moving median residuals
            3. Calculate consensus scores
        
        Args:
            df: DataFrame with temperature clusters and features.
        
        Note:
            Modifies df in-place by adding score columns:
                - 'lof_score': LOF anomaly scores
                - 'residual_score': Moving median residual scores
                - 'consensus_score': Weighted combination of both
        """
        logger.info("="*70)
        logger.info("[Stage 2] Local Anomaly Detection")
        logger.info("="*70)
        
        # Initialize score columns
        df['lof_score'] = 0.0
        df['residual_score'] = 0.0
        df['consensus_score'] = 0.0
        
        n_clusters_processed = 0
        n_clusters_skipped = 0
        
        lof_cols_idx = self.system_defaults['lof_cols_idx']
        lof_col_names = [df.columns[idx] for idx in lof_cols_idx]
        
        for cluster_id in sorted(df['temp_cluster'].unique()):
            cluster_mask = df['temp_cluster'] == cluster_id
            cluster_indices = df[cluster_mask].index
            
            # Extract LOF features
            cluster_lof_data = df.loc[cluster_indices, lof_col_names].values
            cluster_size = len(cluster_lof_data)
            
            # Get temperature range for logging
            temp_values = df.loc[cluster_indices, 'temperature']
            temp_min, temp_max = temp_values.min(), temp_values.max()
            
            if cluster_size <= self.config.min_cluster_size:
                # Skip small clusters
                logger.info(
                    f"  Cluster {cluster_id:2d}: size={cluster_size:4d}, "
                    f"T={temp_min:7.2f}~{temp_max:7.2f}, ⚠️  Skipped (too small)"
                )
                n_clusters_skipped += 1
                continue
            
            # ========== LOF Detection ==========
            actual_n_neighbors = min(cluster_size - 1, self.config.n_neighbors)
            
            lof = LocalOutlierFactor(
                n_neighbors=actual_n_neighbors,
                contamination=self.config.contamination
            )
            lof.fit_predict(cluster_lof_data)
            
            # Extract LOF scores (higher = more anomalous)
            raw_lof_scores = -lof.negative_outlier_factor_
            # LOF = 1.0 is normal, > 1.0 is anomalous
            lof_anomaly_scores = np.maximum(raw_lof_scores - 1.0, 0.0)
            df.loc[cluster_indices, 'lof_score'] = lof_anomaly_scores
            
            # ========== Moving Median Residual Detection ==========
            if self.config.enable_residual_detection:
                residual_scores = self._compute_moving_median_residuals(
                    cluster_lof_data
                )
                df.loc[cluster_indices, 'residual_score'] = residual_scores
                
                # ========== Consensus Scoring ==========
                # Normalize both scores to [0, 1] within cluster
                lof_norm = self._normalize(lof_anomaly_scores)
                residual_norm = self._normalize(residual_scores)
                
                # Weighted combination
                consensus = (
                    self.config.consensus_weight * lof_norm +
                    (1 - self.config.consensus_weight) * residual_norm
                )
                df.loc[cluster_indices, 'consensus_score'] = consensus
                
                # Logging
                lof_mean = lof_norm.mean()
                res_mean = residual_norm.mean()
                con_mean = consensus.mean()
                
                logger.info(
                    f"  Cluster {cluster_id:2d}: size={cluster_size:4d}, "
                    f"T={temp_min:7.2f}~{temp_max:7.2f}, "
                    f"LOF={lof_mean:.3f}, Residual={res_mean:.3f}, "
                    f"Consensus={con_mean:.3f}"
                )
            else:
                # Use only LOF if residual detection is disabled
                df.loc[cluster_indices, 'consensus_score'] = self._normalize(
                    lof_anomaly_scores
                )
                logger.info(
                    f"  Cluster {cluster_id:2d}: size={cluster_size:4d}, "
                    f"T={temp_min:7.2f}~{temp_max:7.2f}, "
                    f"LOF={lof_anomaly_scores.mean():.3f}"
                )
            
            n_clusters_processed += 1
        
        self.info_['n_clusters_processed'] = n_clusters_processed
        self.info_['n_clusters_skipped'] = n_clusters_skipped
    
    def _compute_moving_median_residuals(
        self,
        cluster_data: np.ndarray
    ) -> np.ndarray:
        """Compute moving median residual scores for trend deviation detection.
        
        This method:
            1. Sorts data by first feature dimension
            2. Computes moving median of second feature
            3. Calculates residuals from median trend
            4. Normalizes using MAD (Median Absolute Deviation)
            5. Converts to anomaly scores using exponential transform
        
        Args:
            cluster_data: Feature data for cluster, shape (n_samples, n_features).
        
        Returns:
            Anomaly scores in [0, 1], shape (n_samples,).
        """
        n_samples = len(cluster_data)
        
        # Sort by first feature (e.g., w_comp1 for solubility, pressure for viscosity)
        sort_col_idx = self.system_defaults['residual_sort_col_idx']
        sort_indices = np.argsort(cluster_data[:, sort_col_idx])
        sorted_data = cluster_data[sort_indices]
        
        # Target variable is the second feature
        y = sorted_data[:, 1] if sorted_data.shape[1] > 1 else sorted_data[:, 0]
        
        # Compute moving median
        window_size = min(
            self.config.moving_median_window,
            max(3, n_samples // 3)
        )
        min_periods = max(2, window_size // 2)
        
        smoothed = pd.Series(y).rolling(
            window=window_size,
            center=True,
            min_periods=min_periods
        ).median().values
        
        # Compute residuals
        residuals = np.abs(y - smoothed)
        
        # Normalize using MAD (Median Absolute Deviation)
        mad = np.nanmedian(residuals)
        if mad < 1e-6:
            mad = np.nanstd(residuals)
            if mad < 1e-6:
                mad = 1.0
        
        normalized_residuals = residuals / (mad + 1e-6)
        
        # Convert to anomaly scores using exponential transform
        # score = 1 - exp(-residual/lambda)
        # lambda=3 means 3*MAD -> score≈0.63, 6*MAD -> score≈0.86
        residual_scores_sorted = 1.0 - np.exp(-normalized_residuals / 3.0)
        
        # Restore original order
        residual_scores = np.zeros(n_samples)
        residual_scores[sort_indices] = residual_scores_sorted
        
        return residual_scores
    
    def _filter_outliers(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter outliers using global consensus ranking.
        
        This method:
            1. Globally normalizes consensus scores across all samples
            2. Ranks samples by score (high to low)
            3. Removes top N% as outliers (N = contamination rate)
        
        Args:
            df: DataFrame with consensus scores and original indices.
            X: Original feature matrix.
            y: Original target array.
        
        Returns:
            Tuple of (X_clean, y_clean, outlier_indices).
        """
        logger.info("="*70)
        logger.info("[Stage 3] Global Consensus Ranking and Filtering")
        logger.info("="*70)
        
        # Global normalization of consensus scores
        global_consensus = df['consensus_score'].values
        global_consensus_norm = self._normalize(global_consensus)
        df['global_consensus_norm'] = global_consensus_norm
        
        # Determine number of outliers to remove
        n_total = len(df)
        n_to_remove = int(n_total * self.config.contamination)
        
        if n_to_remove > 0:
            # Sort by score (highest first) and take top N
            sorted_indices = np.argsort(global_consensus_norm)[::-1]
            outlier_df_indices = sorted_indices[:n_to_remove]
            
            # Create clean mask
            is_clean = np.ones(n_total, dtype=bool)
            is_clean[outlier_df_indices] = False
            
            threshold_score = global_consensus_norm[sorted_indices[n_to_remove - 1]]
            logger.info(f"Global consensus threshold: {threshold_score:.4f}")
            logger.info(
                f"Removing top {n_to_remove} points "
                f"({self.config.contamination*100:.1f}%)"
            )
        else:
            is_clean = np.ones(n_total, dtype=bool)
            logger.info("No outliers to remove (contamination rate = 0)")
        
        # Extract clean data
        X_clean = X[is_clean]
        y_clean = y[is_clean]
        
        # Get original indices of removed outliers
        outlier_mask = ~is_clean
        outlier_indices = df.loc[outlier_mask, 'original_index'].to_numpy()
        
        # Update info dictionary
        self.info_['n_outliers'] = len(outlier_indices)
        self.info_['outlier_rate'] = len(outlier_indices) / n_total
        self.info_['iterations'] = 1
        self.info_['method'] = 'TKMeans-LOF-Consensus'
        
        return X_clean, y_clean, outlier_indices
    
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """Min-Max normalize array to [0, 1].
        
        Args:
            arr: Input array.
        
        Returns:
            Normalized array in [0, 1]. Returns zeros if range is too small.
        """
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        
        return (arr - arr_min) / (arr_max - arr_min)


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Provide backward compatible class name
KMeansLOFCleaner = TKMeansLOF


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("TKMeansLOF - Temperature-based K-Means + LOF Outlier Detector")
    print("="*70)
    print("\nRunning tests...\n")
    
    # ========== Test 1: Solubility System ==========
    print("\n" + "="*70)
    print("Test 1: Solubility System (3 columns)")
    print("="*70)
    
    np.random.seed(42)
    n_points = 500
    
    # Generate solubility data: [T, w_comp1, w_comp2]
    T_sol = np.random.uniform(-10, 100, n_points)
    w1_sol = np.random.uniform(0, 50, n_points)
    w2_sol = np.random.uniform(0, 60, n_points)
    X_sol = np.column_stack([T_sol, w1_sol, w2_sol])
    y_sol = np.random.rand(n_points) * 100
    
    # Add outliers
    n_outliers = 50
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    X_sol[outlier_idx] += np.random.randn(n_outliers, 3) * 20
    
    print(f"Original data: {len(X_sol)} points")
    print(f"Artificial outliers: {n_outliers} points\n")
    
    # Test with backward compatible interface
    cleaner_sol = TKMeansLOF(
        config={
            'lof_contamination': 0.20,
            'lof_n_neighbors': 5,
            'auto_select_k': True,
            'min_cluster_size': 10
        },
        system_type='solubility',
        verbose=True
    )
    
    X_clean_sol, y_clean_sol, removed_sol, info_sol = cleaner_sol.clean(X_sol, y_sol)
    
    print("\n" + "="*70)
    print("Solubility Test Results")
    print("="*70)
    print(f"Removed: {len(removed_sol)} ({info_sol['outlier_rate']*100:.2f}%)")
    print(f"Retained: {len(X_clean_sol)} ({len(X_clean_sol)/len(X_sol)*100:.2f}%)")
    print(f"Method: {info_sol['method']}")
    
    # ========== Test 2: Viscosity System ==========
    print("\n\n" + "="*70)
    print("Test 2: Viscosity System (4 columns)")
    print("="*70)
    
    np.random.seed(42)
    n_points = 400
    
    # Generate viscosity data: [T, pressure, comp1, comp2]
    T_vis = np.random.uniform(20, 120, n_points)
    P_vis = np.random.uniform(1e5, 1e7, n_points)
    comp1_vis = np.random.uniform(0, 1, n_points)
    comp2_vis = np.random.uniform(0, 1, n_points)
    X_vis = np.column_stack([T_vis, P_vis, comp1_vis, comp2_vis])
    y_vis = np.random.uniform(0.5, 5.0, n_points)
    
    # Add outliers
    n_outliers = 20
    outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
    X_vis[outlier_idx] += np.random.randn(n_outliers, 4) * [10, 1e6, 0.3, 0.3]
    
    print(f"Original data: {len(X_vis)} points")
    print(f"Artificial outliers: {n_outliers} points\n")
    
    cleaner_vis = TKMeansLOF(
        config={
            'lof_contamination': 0.05,
            'lof_n_neighbors': 10,
            'auto_select_k': True,
            'min_cluster_size': 10
        },
        system_type='viscosity',
        verbose=True
    )
    
    X_clean_vis, y_clean_vis, removed_vis, info_vis = cleaner_vis.clean(X_vis, y_vis)
    
    print("\n" + "="*70)
    print("Viscosity Test Results")
    print("="*70)
    print(f"Removed: {len(removed_vis)} ({info_vis['outlier_rate']*100:.2f}%)")
    print(f"Retained: {len(X_clean_vis)} ({len(X_clean_vis)/len(X_vis)*100:.2f}%)")
    print(f"Method: {info_vis['method']}")
    
    # ========== Interface Validation ==========
    print("\n\n" + "="*70)
    print("Interface Compatibility Validation")
    print("="*70)
    
    try:
        # Test solubility
        assert isinstance(X_clean_sol, np.ndarray), "X_clean should be np.ndarray"
        assert isinstance(y_clean_sol, np.ndarray), "y_clean should be np.ndarray"
        assert isinstance(removed_sol, np.ndarray), "outlier_indices should be np.ndarray"
        assert isinstance(info_sol, dict), "info should be dict"
        assert 'n_outliers' in info_sol, "info should contain 'n_outliers'"
        assert 'outlier_rate' in info_sol, "info should contain 'outlier_rate'"
        assert 'method' in info_sol, "info should contain 'method'"
        assert X_clean_sol.shape[1] == 3, "Solubility X_clean should have 3 columns"
        
        # Test viscosity
        assert X_clean_vis.shape[1] == 4, "Viscosity X_clean should have 4 columns"
        
        print("All interface tests passed!")
        print("Backward compatibility maintained!")
        print("Both systems working correctly!")
        
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    