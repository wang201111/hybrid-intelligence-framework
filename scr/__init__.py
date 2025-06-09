"""
Hybrid Intelligence Framework for Small-Sample Scientific Prediction

A multi-scale physical constraints and data synergy framework integrating:
- T-KMeans-LOF: Temperature clustering-guided outlier detection
- IADAF: Integrated adaptive data augmentation framework
- LDPC: Low-dimensional physical constraints
"""

from .preprocessing import TKMeansLOF
from .augmentation import IADAF
from .constraints import LDPCConstraints, LOESSModel, CurveTransformation
from .framework import HybridIntelligenceFramework, DNNModel

__version__ = "1.0.0"
__author__ = "Authors"

__all__ = [
    "TKMeansLOF",
    "IADAF", 
    "LDPCConstraints",
    "LOESSModel",
    "CurveTransformation",
    "HybridIntelligenceFramework",
    "DNNModel"
]