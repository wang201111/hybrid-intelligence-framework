# Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15624559.svg)](https://doi.org/10.5281/zenodo.15624559)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of the hybrid intelligence framework described in:

**"Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data"**

*Yuan Wang, Tiancheng Wang, Jindi Du, Qian Chen, Shuaiying Yuan, Song Li, Weidong Zhang, Dahuan Liu*

**Status**: Under review at *Nature Communications*

## Abstract

This framework addresses the triple challenge of data noise, data scarcity, and physical inconsistency in small-sample scientific prediction through three synergistic modules:

- **T-KMeans-LOF**: Temperature clustering-guided local outlier detection for data cleaning
- **IADAF**: Integrated Adaptive Data Augmentation Framework using WGAN-GP with Bayesian optimization
- **LDPC**: Low-Dimensional Physical Constraints ensuring thermodynamic consistency

The framework achieves remarkable high-temperature extrapolation improvements:
- KCl-MgCl2-H2O: R² increases from 0.57 to 0.87 (extrapolating from -34~100°C to 100~227°C)
- NaCl-KCl-H2O: R² increases from 0.67 to 0.97 (extrapolating from -23~50°C to 50~180°C)
- MCH-cis-Decalin-SCOS viscosity: R² increases from 0.04 to 0.94 (extrapolating from 20~40°C to 40~80°C)

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Description](#data-description)
- [Usage Examples](#usage-examples)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Framework Architecture](#framework-architecture)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## System Requirements

### Hardware Requirements

**Minimum:**
- CPU: Multi-core processor (2+ cores)
- RAM: 8 GB
- Storage: 1 GB available space

**Recommended:**
- CPU: Intel i5 or equivalent
- RAM: 16 GB or higher
- GPU: NVIDIA GPU with CUDA support (optional, accelerates training by ~4x)
- Storage: 2 GB available space

**Development Environment:**
- Tested on: Intel i5-13600KF (3.5GHz), NVIDIA RTX 4070 Super, 32GB RAM

### Software Requirements

- Python 3.8 or higher (tested on Python 3.9)
- CUDA 11.0+ (optional, for GPU acceleration)

**Tested Operating Systems:**
- Ubuntu 20.04 LTS
- Windows 10/11
- macOS 12+

**Expected Runtime:**
- Single system (5-fold cross-validation): ~30 minutes (GPU) / ~2 hours (CPU)
- Full reproduction (4 systems): ~2 hours (GPU) / ~8 hours (CPU)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/wang201111/hybrid-intelligence-framework.git
cd hybrid-intelligence-framework
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Or using conda
conda create -n physics-ml python=3.9
conda activate physics-ml
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import pandas; import sklearn; print('Installation successful!')"
```

## Quick Start

### Minimal Example

```python
from src.solubility_pipeline import SolubilityPipeline

# Initialize pipeline
pipeline = SolubilityPipeline()

# Load data
pipeline.load_data('data/solubility/raw/ternary/KCl_MgCl2_H2O.xlsx')

# Run complete framework
results = pipeline.run_complete_model()

# Display results
print(f"Test R²: {results['test_r2']:.3f}")
print(f"Physical Rationality: {results['physics_score']:.3f}")
print(f"Boundary Consistency: {results['boundary_consistency']:.3f}")
```

### Expected Output

```
Processing KCl-MgCl2-H2O system...
Training on low-temperature data (-34 to 100°C, 484 points)
Testing on high-temperature data (100 to 227°C, 135 points)

Results:
Test R²: 0.873
Physical Rationality: 0.978
Boundary Consistency: 0.982
```

## Data Description

All experimental data are publicly available and compiled from literature sources (see References 57-69 in the paper). Data files are provided in `.xlsx` format with standardized column names.

### Solubility Systems (3 ternary salt-water systems)

#### 1. KCl-MgCl2-H2O System
- **Total datapoints**: 619
- **Temperature range**: -34°C to 227°C
- **Training set**: -34.5°C to 100°C (484 points)
- **Test set**: 100°C to 227°C (135 points, high-temperature extrapolation)
- **Input features**: [Temperature (°C), w(KCl) (%)]
- **Target variable**: w(MgCl2) (%)

#### 2. NaCl-KCl-H2O System
- **Total datapoints**: 463
- **Temperature range**: -23°C to 180°C
- **Training set**: -23°C to 50°C (316 points)
- **Test set**: 50°C to 180°C (147 points, high-temperature extrapolation)
- **Input features**: [Temperature (°C), w(NaCl) (%)]
- **Target variable**: w(KCl) (%)

#### 3. NaCl-MgCl2-H2O System
- **Total datapoints**: 313
- **Temperature range**: -35°C to 200°C
- **Training set**: -35°C to 100°C (244 points)
- **Test set**: 100°C to 200°C (69 points, high-temperature extrapolation)
- **Input features**: [Temperature (°C), w(NaCl) (%)]
- **Target variable**: w(MgCl2) (%)

### Viscosity System (1 ternary organic solvent system)

#### MCH-cis-Decalin-SCOS System
- **Total datapoints**: 546
- **Temperature range**: 20°C to 80°C
- **Pressure range**: 0.1 to 100 MPa
- **Training set**: 20°C to 40°C (136 points)
- **Test set**: 40°C to 80°C (387 points, high-temperature extrapolation)
- **Input features**: [Temperature (°C), Pressure (MPa), x(MCH), x(cis-Decalin)]
- **Target variable**: Viscosity (mPa·s)

**Note**: MCH = methylcyclohexane; SCOS = 2,2,4,4,6,8,8-heptamethylnonane

### Data Format

All data files follow a consistent format:

**Solubility data** (`*.xlsx`):
```
| Temperature | Composition1 | Composition2 |
|-------------|--------------|--------------|
| -34.5       | 0.0          | 57.51        |
| -10.0       | 5.2          | 48.3         |
| ...         | ...          | ...          |
```

**Viscosity data** (`*.xlsx`):
```
| Temperature | Pressure | x1    | x2    | Viscosity |
|-------------|----------|-------|-------|-----------|
| 20.0        | 0.1      | 0.333 | 0.333 | 1.234     |
| 20.0        | 10.0     | 0.333 | 0.333 | 1.256     |
| ...         | ...      | ...   | ...   | ...       |
```

### Data Directory Structure

```
data/
├── solubility/
│   ├── raw/
│   │   ├── ternary/
│   │   │   ├── KCl_MgCl2_H2O.xlsx
│   │   │   ├── NaCl_KCl_H2O.xlsx
│   │   │   └── NaCl_MgCl2_H2O.xlsx
│   │   └── binary/
│   │       ├── KCl_H2O.xlsx
│   │       ├── MgCl2_H2O.xlsx
│   │       └── NaCl_H2O.xlsx
│   └── split_by_temperature/
│       ├── KCl_MgCl2_H2O_low_temp.xlsx  # Training data
│       ├── KCl_MgCl2_H2O_high_temp.xlsx # Testing data
│       └── ...
└── viscosity/
    ├── raw/
    │   ├── ternary/
    │   │   └── MCH_cis_Decalin_HMN.xlsx
    │   └── binary/
    │       ├── MCH_cis_Decalin.xlsx
    │       ├── MCH_HMN.xlsx
    │       └── cis_Decalin_HMN.xlsx
    └── split_by_temperature/
        ├── low_temp.xlsx   # Training data (20-40°C)
        └── high_temp.xlsx  # Testing data (40-80°C)
```

## Usage Examples

### Example 1: Data Cleaning with T-KMeans-LOF

```python
from src.t_kmeans_lof import TKMeansLOF
import pandas as pd

# Load raw data
raw_data = pd.read_excel('data/solubility/raw/ternary/KCl_MgCl2_H2O.xlsx')

# Initialize outlier detector
detector = TKMeansLOF(
    n_clusters='auto',           # Automatic cluster selection
    contamination=0.20,          # Expected outlier proportion
    k_neighbors=5,               # LOF parameter
    window_size=5                # RMM window size
)

# Detect and remove outliers
cleaned_data, outlier_mask = detector.fit_predict(raw_data)

print(f"Original samples: {len(raw_data)}")
print(f"Cleaned samples: {len(cleaned_data)}")
print(f"Removed outliers: {outlier_mask.sum()}")
```

### Example 2: Data Augmentation with IADAF

```python
from src.iadaf import IADAFTrainer
import pandas as pd

# Load cleaned data
cleaned_data = pd.read_excel('data/solubility/cleaned/KCl_MgCl2_H2O_cleaned.xlsx')

# Initialize IADAF trainer
augmentor = IADAFTrainer(
    n_synthetic=1500,            # Number of synthetic samples
    latent_dim_range=(1, 100),   # Bayesian optimization range
    hidden_dim_range=(150, 350), # Bayesian optimization range
    n_iterations=100             # Bayesian optimization iterations
)

# Generate synthetic data
synthetic_data = augmentor.fit_generate(cleaned_data)

# Combine real and synthetic data
augmented_data = pd.concat([cleaned_data, synthetic_data], ignore_index=True)

print(f"Original samples: {len(cleaned_data)}")
print(f"Synthetic samples: {len(synthetic_data)}")
print(f"Total samples: {len(augmented_data)}")
```

### Example 3: Apply Physical Constraints (LDPC)

```python
from src.ldpc_solubility import ConformingCorrectedModel
from src.binary_predictor import BinaryPredictor
import torch

# Train base DNN model
base_model = torch.nn.Sequential(
    torch.nn.Linear(2, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

# Train binary boundary models
binary_models = {
    'left': BinaryPredictor('data/solubility/raw/binary/MgCl2_H2O.xlsx'),
    'right': BinaryPredictor('data/solubility/raw/binary/KCl_H2O.xlsx')
}

for name, model in binary_models.items():
    model.train()

# Initialize LDPC model
ldpc_model = ConformingCorrectedModel(
    base_model=base_model,
    binary_models=binary_models,
    decay_rate=2.0               # Exponential decay parameter
)

# Make predictions with physical constraints
X_test = torch.tensor([[150.0, 10.5]])  # [Temperature, w(KCl)]
predictions = ldpc_model.predict(X_test)

print(f"Predicted w(MgCl2): {predictions[0]:.2f}%")
```

### Example 4: Complete End-to-End Pipeline

```python
from src.solubility_pipeline import SolubilityPipeline

# Initialize pipeline with custom configuration
pipeline = SolubilityPipeline(
    contamination=0.20,
    n_synthetic=1500,
    k_folds=5
)

# Load data
pipeline.load_data('data/solubility/raw/ternary/KCl_MgCl2_H2O.xlsx')

# Run ablation experiments
results = {}

# 1. Baseline (DNN only)
results['baseline'] = pipeline.run_baseline()

# 2. T-KMeans-LOF only
results['t_kmeans_lof'] = pipeline.run_with_cleaning_only()

# 3. IADAF only
results['iadaf'] = pipeline.run_with_augmentation_only()

# 4. LDPC only
results['ldpc'] = pipeline.run_with_constraints_only()

# 5. Complete framework
results['complete'] = pipeline.run_complete_model()

# Compare results
import pandas as pd
comparison = pd.DataFrame({
    name: {
        'Test R²': res['test_r2'],
        'Physics Score': res['physics_score'],
        'Boundary Consistency': res['boundary_consistency']
    }
    for name, res in results.items()
}).T

print(comparison)
```

## Reproducing Paper Results

### Reproduce Table 2 (Main Results)

```bash
python examples/reproduce_table2.py
```

**Expected output:**

```
================================================================================
TABLE 2 REPRODUCTION - Unified Performance Summary
================================================================================

System              | Baseline Test R²  | Complete Test R²  | Improvement
--------------------|-------------------|-------------------|-------------
KCl-MgCl2-H2O      | 0.572 ± 0.313     | 0.873 ± 0.015     | +0.301
NaCl-KCl-H2O       | 0.674 ± 0.319     | 0.972 ± 0.004     | +0.298
NaCl-MgCl2-H2O     | 0.776 ± 0.147     | 0.817 ± 0.060     | +0.041
MCH-cis-Decalin    | 0.042 ± 0.713     | 0.936 ± 0.026     | +0.894

Physical Rationality Scores:
KCl-MgCl2-H2O      | 0.612 → 0.978
NaCl-KCl-H2O       | 0.672 → 0.955
NaCl-MgCl2-H2O     | 0.711 → 0.978
MCH-cis-Decalin    | 0.595 → 0.833
```

### Reproduce Figure 2 (Outlier Detection)

```bash
python examples/reproduce_figure2.py
```

### Reproduce Figure 5 (Ablation Study)

```bash
python examples/reproduce_figure5.py
```

### Run Small-Sample Robustness Test

```bash
python examples/small_sample_robustness.py --sample_ratios 0.1 0.25 0.5 0.75 1.0
```

### Run Noise Robustness Test

```bash
python examples/noise_robustness.py --noise_levels 0.05 0.10 0.15 0.20
```

## Framework Architecture

### Core Modules

The framework consists of three core modules implemented in `src/`:

#### 1. T-KMeans-LOF (`src/t_kmeans_lof.py`)
- **Purpose**: Temperature-guided outlier detection
- **Key Classes**:
  - `TKMeansLOF`: Main outlier detector
  - `OutlierDetectionConfig`: Configuration dataclass
- **Algorithm**: Combines K-means clustering on temperature with Local Outlier Factor (LOF) and Robust Moving Median (RMM)

#### 2. IADAF (`src/iadaf.py`)
- **Purpose**: Adaptive data augmentation
- **Key Classes**:
  - `IADAFTrainer`: Main augmentation framework
  - `Generator`: WGAN-GP generator network
  - `Discriminator`: WGAN-GP discriminator network
- **Algorithm**: Wasserstein GAN with Gradient Penalty + Bayesian hyperparameter optimization

#### 3. LDPC (`src/ldpc_solubility.py`, `src/ldpc_viscosity.py`)
- **Purpose**: Low-dimensional physical constraints
- **Key Classes**:
  - `ConformingCorrectedModel`: Solubility physical constraints
  - `EdgeSamplingCorrectedModel`: Viscosity physical constraints
- **Algorithm**: Exponential decay-weighted boundary correction

### Pipeline Modules

#### Solubility Pipeline (`src/solubility_pipeline.py`)
- Integrates all three modules for solubility prediction
- Supports 5-fold cross-validation
- Handles temperature-based train/test splitting

#### Viscosity Pipeline (`src/viscosity_pipeline.py`)
- Integrates all three modules for viscosity prediction
- Handles 4D input space (T, P, x1, x2)
- Supports three-boundary constraints

### Utility Modules

#### Physics Evaluators (`src/utils_solubility.py`, `src/utils_viscosity.py`)
- **Boundary Consistency**: Measures convergence to binary subsystem boundaries
- **Thermodynamic Smoothness**: Evaluates Laplacian-based surface smoothness
- **Physical Rationality**: Combined score of boundary consistency and smoothness

### Binary Predictors (`src/binary_predictor.py`)
- Trains ensemble models for binary subsystem boundaries
- Supports both solubility and viscosity systems
- Implements bootstrap aggregating for robust predictions

## Framework Design Principles

1. **Modularity**: Each component (cleaning, augmentation, constraints) functions independently
2. **Synergy**: Components are designed to complement each other's limitations
3. **Transferability**: Framework applies to both thermodynamic (solubility) and transport (viscosity) properties
4. **Physical Consistency**: All predictions satisfy fundamental thermodynamic laws

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2025physics,
  title={Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data},
  author={Wang, Yuan and Wang, Tiancheng and Du, Jindi and Chen, Qian and Yuan, Shuaiying and Li, Song and Zhang, Weidong and Liu, Dahuan},
  journal={Nature Communications},
  year={2025},
  note={Under review}
}
```

### Related Publications

This work builds upon established methodologies in physics-informed machine learning:

- Raissi et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
- Cai et al. (2021). Physics-informed neural networks for fluid mechanics. *Acta Mechanica Sinica*, 37, 1727-1738.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure your code:
- Follows PEP 8 style guidelines
- Includes appropriate docstrings
- Passes all existing tests
- Includes new tests for new features

## Troubleshooting

### Common Issues

**Issue 1: ImportError for PyTorch**
```bash
# Solution: Install PyTorch with appropriate CUDA version
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

**Issue 2: XLSX file reading errors**
```bash
# Solution: Ensure openpyxl is installed
pip install openpyxl
```

**Issue 3: Out of memory during training**
```bash
# Solution: Reduce batch size or use CPU instead of GPU
python examples/reproduce_table2.py --device cpu --batch_size 16
```

**Issue 4: Slow training on CPU**
```bash
# Solution: Use GPU if available or reduce cross-validation folds
python examples/reproduce_table2.py --device cuda --k_folds 3
```

## Frequently Asked Questions

**Q: Can I apply this framework to my own dataset?**
A: Yes! The framework is designed to be generalizable. Ensure your data follows the format described in [Data Description](#data-description) and you have corresponding binary system data for physical constraints.

**Q: What if I don't have binary system data?**
A: You can run the framework without LDPC constraints by using `pipeline.run_without_constraints()`. However, physical rationality may be compromised.

**Q: How do I choose the contamination parameter?**
A: The default value of 0.20 works well for multi-source experimental data. For single-source high-quality data, try 0.05-0.10. For very noisy data, try 0.25-0.30.

**Q: Can I use this for classification tasks?**
A: The current implementation is designed for regression. Adapting it to classification would require modifications to the loss functions and evaluation metrics.

## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (22478015)
- Qinghai University Research Ability Enhancement Project (2025KTST02)

We thank the reviewers at Nature Communications for their valuable feedback in improving this work.

## Contact

For questions, bug reports, or collaboration inquiries:

**Corresponding Authors:**
- Dahuan Liu: liudh@mail.buct.edu.cn
- Weidong Zhang: weidzhang1208@126.com

**Code Maintainer:**
- Yuan Wang: wang201111@github.com

**Issue Tracker:** https://github.com/wang201111/hybrid-intelligence-framework/issues

**Response Time:** Typically within 48 hours

## Version History

- **v1.0.0** (2025-01): Initial release for Nature Communications submission
  - Complete implementation of T-KMeans-LOF, IADAF, and LDPC
  - Validation on 3 solubility systems and 1 viscosity system
  - Full reproduction scripts for paper results

---

**Last Updated:** January 2025  
**Repository:** https://github.com/wang201111/hybrid-intelligence-framework  
**DOI:** https://doi.org/10.5281/zenodo.15624559
