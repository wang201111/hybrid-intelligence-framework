# Physics-Constrained Multi-Module Machine Learning Framework for Solubility Prediction from Small-Sample Experimental Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15624558.svg)](https://doi.org/10.5281/zenodo.15624558)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of the framework described in the paper:

**"Physics-Constrained Multi-Module Machine Learning Framework for Solubility Prediction from Small-Sample Experimental Data"**

*Submitted to Nature Communications*

The framework addresses three critical challenges in small-sample scientific prediction: data noise, data scarcity, and physical inconsistency. It integrates three complementary modules:

- **T-KMeans-LOF**: Temperature-guided outlier detection based on thermodynamic stability principle
- **IADAF**: Iterative adaptive data augmentation using WGAN-GP with Bayesian optimization
- **LDPC**: Low-dimensional physical constraints for boundary consistency

Unlike traditional PINNs that require explicit governing equations, this framework embeds macroscopic physical principles as soft constraints, enabling broad applicability across diverse chemical systems.

## Key Features

- High-temperature extrapolation from limited training data
- Physics-informed constraints without explicit differential equations
- Robust performance under data scarcity and noise contamination
- Modular design allowing independent evaluation of each component

## System Requirements

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Tested Environments

**Operating Systems:**
- Ubuntu 20.04 / 22.04 LTS
- Windows 10/11
- macOS 12+ (CPU only, no CUDA support)

**Python Versions:**
- Python 3.8, 3.9, 3.10 (recommended: 3.9)

**GPU/CUDA:**
- NVIDIA RTX 4070 Super (tested)
- NVIDIA RTX 3080 (tested)
- CUDA 11.7 / 11.8
- cuDNN 8.5+

**Note**: CPU-only mode is supported but significantly slower (~10x).

## Installation

### Setup

Clone the repository:
```bash
git clone https://github.com/wang201111/physics-constrained-solubility-framework.git
cd physics-constrained-solubility-framework
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Installation Time

On a typical desktop computer with stable internet:
- **Dependencies installation**: ~5-10 minutes (via pip)
- **Data download**: Included in repository (~50 MB)
- **Total setup time**: ~10-15 minutes

## Quick Start

```python
from src.solubility_pipeline import SolubilityPipeline

# Initialize complete framework
pipeline = SolubilityPipeline(
    system_name='KCl-MgCl2-H2O',
    use_outlier_detection=True,
    use_augmentation=True,
    use_physical_constraints=True
)

# Load data
pipeline.load_data('data/solubility/raw/ternary/KCl_MgCl2_H2O.xlsx')

# Run workflow
results = pipeline.run()

print(f"Test R2: {results['test_r2']:.3f}")
print(f"Boundary Consistency: {results['boundary_consistency']:.3f}")
```

## How to Run

All experiments are standalone Python scripts that can be executed directly from the project root.

### Quick Start

Run a single experiment:
```bash
# Ablation study - Complete Model (M4 pipeline)
python experiments/solubility/ablation/Complete_Model_experiment.py

# Ablation study - Baseline
python experiments/solubility/ablation/baseline_experiment.py
```

### Ablation Studies

Compare the contribution of each module:
```bash
cd experiments/solubility/ablation

python baseline_experiment.py              # Baseline DNN
python T_KMeans_LOF_experiment.py          # + Data cleaning
python IADAF_experiment.py                 # + Data augmentation
python LDPC_experiment.py                  # + Physical constraints
python Complete_Model_experiment.py        # All modules combined
```

**Runtime**: ~5 minutes each (Complete Model: ~40 minutes)

### Small-Sample Robustness Tests

Evaluate performance with reduced training data (10%, 25%, 50%, 75%, 100%):
```bash
cd experiments/solubility/small_sample

python small_sample_baseline_experiment.py
python small_sample_T_KMeans_LOF_experiment.py
python small_sample_IADAF_experiment.py
python small_sample_LDPC_experiment.py
python small_sample_Complete_Model_experiment.py
```

**Runtime**: ~30 minutes each (Complete Model: ~8 hours)

### Noise Robustness Tests

Test tolerance to measurement errors (0%, 5%, 10%, 15%, 20% noise):
```bash
cd experiments/solubility/noise

python noise_robustness_baseline_experiment.py
python noise_robustness_T_KMeans_LOF_experiment.py
python noise_robustness_IADAF_experiment.py
python noise_robustness_LDPC_experiment.py
python noise_robustness_Complete_Model_experiment.py
```

**Runtime**: ~30 minutes each (Complete Model: ~8 hours)

### Run All Experiments
```bash
cd experiments/solubility

# Ablation studies (~1 hour)
for script in ablation/*.py; do python "$script"; done

# Small-sample tests (~10 hours)
for script in small_sample/*.py; do python "$script"; done

# Noise robustness tests (~10 hours)
for script in noise/*.py; do python "$script"; done
```

## Reproducing Paper Results

### Main Results (Table 2)

Expected results for **KCl-MgCl2-H2O** system:

| Model | Test R² | Test RMSE | Physics Score |
|:------|:--------|:----------|:--------------|
| Baseline | 0.572 ± 0.313 | 8.833 ± 3.460 | 0.612 ± 0.123 |
| Complete Model | 0.873 ± 0.015 | 5.090 ± 0.304 | 0.978 ± 0.002 |

**Training Range**: -34°C to 100°C (484 points)  
**Testing Range**: 100°C to 227°C (135 points, high-temperature extrapolation)

**Run commands:**
```bash
python experiments/solubility/ablation/baseline_experiment.py
python experiments/solubility/ablation/Complete_Model_experiment.py
```

**Expected output:**
```
Training: -34°C to 100°C (484 points)
Testing: 100°C to 227°C (135 points)
==========================================
Baseline Model Test R²: 0.572 ± 0.313
Complete Model Test R²: 0.873 ± 0.015
Improvement: +0.301
```

**Verification:** Results should match within ±0.02 for R² due to random initialization.

### Ablation Studies (Table 3 and Figure 5)

Run all five experimental configurations:
```bash
cd experiments/solubility/ablation

python baseline_experiment.py          # Experiment 1: Baseline (pure DNN on raw data)
python T_KMeans_LOF_experiment.py      # Experiment 2: Data cleaning only
python IADAF_experiment.py             # Experiment 3: Data augmentation only
python LDPC_experiment.py              # Experiment 4: Physical constraints only
python Complete_Model_experiment.py    # Experiment 5: Complete framework (all modules)
```

**Runtime**: Approximately 30 minutes per experiment (except Complete Model: ~40 minutes).

**Compare results:** Check `results/solubility/ablation/*/summary/summary_metrics.xlsx`

### Small-Sample Robustness Tests

Test framework performance with reduced training data:
```bash
cd experiments/solubility/small_sample

python small_sample_baseline_experiment.py
python small_sample_T_KMeans_LOF_experiment.py
python small_sample_IADAF_experiment.py
python small_sample_LDPC_experiment.py
python small_sample_Complete_Model_experiment.py
```

**Expected behavior:**
- Complete Model maintains Test R² > 0.4 even at 10% data (32 samples)
- Baseline Model drops to Test R² < -1.0 at 10% data

### Noise Robustness Tests

Test framework tolerance to measurement errors:
```bash
cd experiments/solubility/noise

python noise_robustness_baseline_experiment.py
python noise_robustness_T_KMeans_LOF_experiment.py
python noise_robustness_IADAF_experiment.py
python noise_robustness_LDPC_experiment.py
python noise_robustness_Complete_Model_experiment.py
```

**Expected behavior:**
- Complete Model maintains Test R² > 0.7 even at 20% noise
- Baseline Model drops to Test R² < -10.0 at 20% noise

## Results Organization

Each experiment uses 5-fold cross-validation. Results are organized as follows:
```
results/solubility/ablation/Complete_Model_results/
├── fold_0/
│   ├── excel/
│   │   ├── complete_metrics.xlsx              # Metrics: R², RMSE, MAE, physics scores
│   │   └── complete_*_predictions.xlsx        # Predictions for train/val/test sets
│   ├── configs/experiment_config.json         # Hyperparameters & random seeds
│   └── models/complete_model.pth              # Trained PyTorch model
├── fold_1/ ... fold_4/                         # Additional folds (same structure)
└── summary/
    ├── summary_metrics.xlsx                    # Aggregated: mean ± std across folds
    ├── test_predictions_summary.xlsx           # Combined predictions from all folds
    └── summary_report.txt                      # Full experimental report
```

### Output Files by Experiment

| Experiment | File Prefix | Example |
|:-----------|:------------|:--------|
| Baseline | `baseline_` | `baseline_metrics.xlsx` |
| T-KMeans-LOF | `cleaning_` | `cleaning_metrics.xlsx` |
| IADAF | `iadaf_` | `iadaf_metrics.xlsx` |
| LDPC | `ldpc_` | `ldpc_metrics.xlsx` |
| Complete Model | `complete_` | `complete_metrics.xlsx` |

**Key Metrics** (in `summary_metrics.xlsx`):
- Statistical: Train/Val/Test R², RMSE, MAE
- Physics: Overall Score, Boundary Consistency, Thermodynamic Smoothness

## Repository Structure

```
physics-informed-ml-framework/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── solubility/
│       ├── raw/
│       │   ├── ternary/
│       │   │   ├── KCl_MgCl2_H2O.xlsx
│       │   │   ├── NaCl_KCl_H2O.xlsx
│       │   │   └── NaCl_MgCl2_H2O.xlsx
│       │   └── binary/
│       │       ├── KCl_H2O.xlsx
│       │       ├── MgCl2_H2O.xlsx
│       │       └── NaCl_H2O.xlsx
│       ├── cleaned/
│       │   ├── KCl_MgCl2_H2O_cleaned.xlsx
│       │   ├── NaCl_KCl_H2O_cleaned.xlsx
│       │   └── NaCl_MgCl2_H2O_cleaned.xlsx
│       ├── split_by_temperature/
│       │   ├── KCl_MgCl2_H2O_low_temp.xlsx
│       │   ├── KCl_MgCl2_H2O_high_temp.xlsx
│       │   ├── NaCl_KCl_H2O_low_temp.xlsx
│       │   ├── NaCl_KCl_H2O_high_temp.xlsx
│       │   ├── NaCl_MgCl2_H2O_low_temp.xlsx
│       │   └── NaCl_MgCl2_H2O_high_temp.xlsx
│       └── fixed_splits/
│           ├── train.xlsx
│           ├── val.xlsx
│           └── test.xlsx
│
├── src/
│   ├── __init__.py
│   ├── binary_predictor.py
│   ├── t_kmeans_lof.py
│   ├── iadaf.py
│   ├── ldpc_solubility.py
│   ├── solubility_pipeline.py
│   └── utils_solubility.py
│
├── models/
│   └── solubility/
│       └── binary/
│           ├── KCl_H2O.pth
│           ├── MgCl2_H2O.pth
│           └── NaCl_H2O.pth
│
├── experiments/
│   └── solubility/
│       ├── __init__.py
│       ├── ablation/
│       │   ├── __init__.py
│       │   ├── baseline_experiment.py
│       │   ├── T_KMeans_LOF_experiment.py
│       │   ├── IADAF_experiment.py
│       │   ├── LDPC_experiment.py
│       │   └── Complete_Model_experiment.py
│       ├── small_sample/
│       │   ├── __init__.py
│       │   ├── small_sample_baseline_experiment.py
│       │   ├── small_sample_T_KMeans_LOF_experiment.py
│       │   ├── small_sample_IADAF_experiment.py
│       │   ├── small_sample_LDPC_experiment.py
│       │   └── small_sample_Complete_Model_experiment.py
│       └── noise/
│           ├── __init__.py
│           ├── noise_robustness_baseline_experiment.py
│           ├── noise_robustness_T_KMeans_LOF_experiment.py
│           ├── noise_robustness_IADAF_experiment.py
│           ├── noise_robustness_LDPC_experiment.py
│           └── noise_robustness_Complete_Model_experiment.py
│
└── results/
    └── solubility/
        ├── ablation/
        │   ├── baseline_results/
        │   ├── T_KMeans_LOF_results/
        │   ├── IADAF_results/
        │   ├── LDPC_results/
        │   └── Complete_Model_results/
        ├── small_sample/
        │   ├── small_sample_baseline_results/
        │   ├── small_sample_T_KMeans_LOF_results/
        │   ├── small_sample_IADAF_results/
        │   ├── small_sample_LDPC_results/
        │   └── small_sample_Complete_Model_results/
        └── noise/
            ├── noise_robustness_baseline_results/
            ├── noise_robustness_T_KMeans_LOF_results/
            ├── noise_robustness_IADAF_results/
            ├── noise_robustness_LDPC_results/
            └── noise_robustness_Complete_Model_results/
```

## Data Description

### Solubility Systems

Three ternary salt-water subsystems of the NaCl-KCl-MgCl2-H2O quaternary system:

**KCl-MgCl2-H2O**
- Temperature range: -34°C to 227°C
- Total datapoints: 619
- Training: -34°C to 100°C (484 points)
- Testing: 100°C to 227°C (135 points, high-temperature extrapolation)

**NaCl-KCl-H2O**
- Temperature range: -23°C to 180°C
- Total datapoints: 463
- Training: -23°C to 50°C (316 points)
- Testing: 50°C to 180°C (147 points, high-temperature extrapolation)

**NaCl-MgCl2-H2O**
- Temperature range: -35°C to 200°C
- Total datapoints: 313
- Training: -35°C to 100°C (244 points)
- Testing: 100°C to 200°C (69 points, high-temperature extrapolation)

### Data Organization

**raw/**: Original experimental data from literature

**cleaned/**: Data after T-KMeans-LOF outlier detection

**split_by_temperature/**: Temperature-based train/test splits
- Low-temperature data: Used for k-fold cross-validation during training
- High-temperature data: Reserved for evaluating extrapolation capability

**fixed_splits/**: Fixed train/val/test splits for noise and small-sample experiments

### Data Format

**Solubility data** (Excel format):
- Temperature (°C)
- Component 1 mass fraction (%)
- Component 2 mass fraction (%)
- Solubility (target variable)

## Framework Components

### Module 1: T-KMeans-LOF

Temperature-guided outlier detection based on the thermodynamic stability principle.

**Key principle**: Equilibrium properties must vary smoothly with state parameters due to analyticity of microscopic interaction potentials and Cahn-Hilliard energy minimization.

**Methodology**:
- K-means clustering stratifies data into near-isothermal layers
- Local Outlier Factor (LOF) detects density-based anomalies within each cluster
- Robust Moving Median (RMM) identifies deviations from thermodynamic smoothness
- Consensus scoring combines both methods to balance detection sensitivity

**Key parameters**:
- `contamination`: Expected proportion of outliers (default: 0.2)
- `n_clusters`: Number of temperature clusters (default: determined by silhouette analysis)
- `n_neighbors`: LOF neighbors parameter (default: 5)

### Module 2: IADAF

Iterative adaptive data augmentation using WGAN-GP with Bayesian optimization.

**Key principle**: Learn intrinsic distribution of cleaned data to generate synthetic samples, transforming discrete experimental points into a continuous data manifold.

**Methodology**:
- WGAN-GP generates synthetic samples with Wasserstein distance and gradient penalty
- Bayesian optimization automatically tunes hyperparameters for each system
- XGBoost discriminator filters generated samples for quality control
- DNN validation R² serves as ultimate quality evaluator

**Optimized hyperparameters**:
- `latent_dim`: Latent space dimensionality (range: 1–100)
- `hidden_dim`: Hidden layer dimensionality (range: 150–350)
- `lambda_gp`: Gradient penalty coefficient (range: 0.05–5.0)

### Module 3: LDPC

Low-dimensional physical constraints based on topological continuity of macroscopic material systems.

**Key principle**: Multicomponent systems must degenerate to corresponding low-dimensional subsystems at composition boundaries.

**Methodology**:
- Binary subsystem models predict boundary values at each temperature
- Dynamic correction during inference (not embedded in training loss)
- Exponential decay weighting ensures boundary accuracy while preserving interior predictions
- Boundary constraints propagate to interior points through interpolation

**Solubility systems**: Two binary boundaries (Component 1–Water, Component 2–Water)

**Key parameters**:
- `decay_rate`: Controls extent of boundary correction (default: k=2)
- Binary models: Ensemble learning with Bootstrap sampling

## Performance Metrics

### Statistical Metrics

- **R² (Coefficient of Determination)**: Model fit quality
- **RMSE (Root Mean Square Error)**: Prediction accuracy
- **MAE (Mean Absolute Error)**: Average deviation

### Physics-Based Metrics

**Physical Rationality**: Overall compliance with thermodynamic laws (mean of Boundary Consistency and Thermodynamic Smoothness)

**Boundary Consistency**: Convergence accuracy to binary subsystem boundaries, measured using normalized RMSE at compositional limits (w(Component) = 0)

**Thermodynamic Smoothness**: Surface continuity based on Cahn-Hilliard theory, quantified via P99 quantile of extreme Laplacian values to penalize non-physical oscillations

## Key Dependencies

Core packages with exact versions specified in `requirements.txt`:

- PyTorch 1.12.0 (deep learning framework)
- NumPy 1.21.0 (numerical computing)
- pandas 1.3.0 (data manipulation)
- scikit-learn 1.0.0 (machine learning utilities)
- openpyxl 3.0.0 (Excel file handling)
- bayesian-optimization 1.2.0 (hyperparameter tuning)
- XGBoost (quality screening in IADAF)

All dependencies are pinned to exact versions to ensure reproducibility.

## Computational Requirements

### Hardware

- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support and 16GB+ RAM
- **Storage**: Approximately 1 GB for code, data, and results

### Runtime

Approximate execution times on **Intel i5-13600KF CPU** with **NVIDIA RTX 4070 Super GPU** and **32GB RAM**:

#### Ablation Studies

| Experiment | Runtime |
|:-----------|:--------|
| Single configuration (non-Complete) | ~5 minutes |
| Complete Model (M4 pipeline) | ~40 minutes |
| All 5 ablation configurations | ~1 hour |

#### Robustness Tests

| Test Type | Single Configuration | Complete Model (M4) |
|:----------|:--------------------|:--------------------|
| Small-sample robustness | ~30 minutes | ~8 hours |
| Noise robustness | ~30 minutes | ~8 hours |

#### Full Experimental Suite

- **Ablation studies** (5 configurations): ~1 hour
- **Small-sample tests** (5 configurations × 5 gradients): ~10 hours
- **Noise tests** (5 configurations × 5 levels): ~10 hours
- **Total**: ~21 hours

> **Note**: Runtime varies based on number of K-fold iterations (default: 5), number of repeats for robustness tests (default: 10–20), IADAF Bayesian optimization iterations (default: 100), and early stopping in DNN training.

## Citation

If you use this code in your research, please cite:
```bibtex
@article{wang2024physics,
  title={Physics-Constrained Multi-Module Machine Learning Framework for Solubility Prediction from Small-Sample Experimental Data},
  author={Wang, Yuan and Wang, Tiancheng and Du, Jindi and Chen, Qian and Zhang, Weidong and Liu, Dahuan},
  journal={Nature Communications},
  year={2024},
  note={Under review}
}
```

## Code Availability

The source code is permanently archived on Zenodo: https://doi.org/10.5281/zenodo.15624558

For the latest updates, visit: https://github.com/wang201111/physics-constrained-solubility-framework

## License

This project is licensed under the MIT License.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact corresponding authors:
  - Prof. Dahuan Liu: liudh@mail.buct.edu.cn
  - Prof. Weidong Zhang: weidzhang1208@126.com

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (22478015) and Qinghai University Research Ability Enhancement Project (2025KTST02).

## Version History

- v1.0.0 (January 2025): Initial release accompanying Nature Communications submission
