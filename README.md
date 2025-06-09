[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15624559.svg)](https://doi.org/10.5281/zenodo.15624559)

# Multi-Scale Physical Constraints and Data Synergy Hybrid Intelligence Framework
# Multi-Scale Physical Constraints and Data Synergy Hybrid Intelligence Framework

This repository contains the implementation of the hybrid intelligence framework proposed in our paper "Multi-Scale Physical Constraints and Data Synergy Hybrid Intelligence Framework: A New Paradigm for Scientific Prediction of Multi-Component Systems Under Small-Sample Conditions".

## Overview

The framework integrates multi-scale physical constraints with data-driven approaches to address challenges in small-sample scientific prediction:

- **T-KMeans-LOF**: Temperature clustering-guided local outlier detection
- **IADAF**: Integrated Adaptive Data Augmentation Framework using WGAN-GP
- **LDPC**: Low-Dimensional Physical Constraints

## Requirements

```bash
numpy==1.21.0
pandas==1.3.0
torch==1.9.0
scikit-learn==1.0.1
scipy==1.7.0
xgboost==1.5.0
joblib==1.1.0
bayes-opt==1.2.0
statsmodels==0.12.0
```

## Installation

```bash
git clone https://github.com/[username]/hybrid-intelligence-framework.git
cd hybrid-intelligence-framework
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing with T-KMeans-LOF

```python
from src.preprocessing import TKMeansLOF

# Initialize the outlier detector
detector = TKMeansLOF(n_clusters='auto', contamination=0.20)

# Fit and transform data
cleaned_data = detector.fit_transform(raw_data)
```

### 2. Data Augmentation with IADAF

```python
from src.augmentation import IADAF

# Initialize the augmentation framework
augmentor = IADAF(
    latent_dim_range=(10, 256),
    hidden_dim_range=(10, 512),
    n_samples=1500
)

# Generate synthetic data
augmented_data = augmentor.fit_generate(cleaned_data)
```

### 3. Apply Physical Constraints

```python
from src.constraints import LDPCConstraints

# Initialize constraints with binary system data
constraints = LDPCConstraints(
    binary_data_path='data/binary_systems/'
)

# Apply constraints to predictions
constrained_predictions = constraints.apply(predictions, temperature)
```

## Data Structure

```
data/
├── raw/                 # Original experimental data
├── processed/           # Cleaned data after T-KMeans-LOF
├── augmented/          # Data after IADAF augmentation
└── binary_systems/     # Binary system solubility data for constraints
```

## Citation

If you use this code in your research, please cite:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please contact: [3152303762@qq.com]
