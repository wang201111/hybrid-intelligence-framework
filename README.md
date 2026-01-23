# Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15624558.svg)](https://doi.org/10.5281/zenodo.15624559)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the complete implementation of the framework described in the paper:

**"Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data"**

*Submitted to Nature Communications*

The framework addresses three critical challenges in small-sample scientific prediction: data noise, data scarcity, and physical inconsistency. It integrates three complementary modules:

- **T-KMeans-LOF**: Temperature-guided outlier detection based on thermodynamic stability principle
- **IADAF**: Iterative adaptive data augmentation using WGAN-GP with Bayesian optimization
- **LDPC**: Low-dimensional physical constraints for boundary consistency

Unlike traditional PINNs that require explicit governing equations, this framework embeds macroscopic physical principles as soft constraints, enabling broad applicability across diverse chemical systems.

## Key Features

- High-temperature extrapolation from limited training data
- Physics-informed constraints without explicit differential equations
- Transferable across different property types (equilibrium and transport properties)
- Robust performance under data scarcity and noise contamination
- Modular design allowing independent evaluation of each component

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Setup

Clone the repository:

```bash
git clone https://github.com/wang201111/hybrid-intelligence-framework.git
cd hybrid-intelligence-framework
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

## Quick Start

### Solubility Prediction Example

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

### Viscosity Prediction Example

```python
from src.viscosity_pipeline import ViscosityPipeline

# Initialize complete framework
pipeline = ViscosityPipeline(
    system_name='MCH-cis-Decalin-HMN',
    use_outlier_detection=True,
    use_augmentation=True,
    use_physical_constraints=True
)

# Load data
pipeline.load_data('data/viscosity/raw/ternary/MCH_cis_Decalin_HMN.xlsx')

# Run workflow
results = pipeline.run()

print(f"Test R2: {results['test_r2']:.3f}")
print(f"Physical Rationality: {results['physics_score']:.3f}")
```

## Reproducing Paper Results

### Main Results (Table 2)

Run the complete framework experiments:

```bash
# Solubility systems
python experiments/solubility/ablation/Complete_Model_experiment.py

# Viscosity system
python experiments/viscosity/ablation/Complete_Model_experiment.py
```

Expected output for KCl-MgCl2-H2O system:
```
Training: -34°C to 100°C (484 points)
Testing: 100°C to 227°C (135 points)
==========================================
Baseline Model Test R2: 0.572 ± 0.313
Complete Model Test R2: 0.873 ± 0.015
Improvement: +0.301
```

### Ablation Studies (Table 3 and Figure 5)

Run all five experimental configurations:

```bash
cd experiments/solubility/ablation

# Experiment 1: Baseline (pure DNN on raw data)
python baseline_experiment.py

# Experiment 2: Data cleaning only
python T_KMeans_LOF_experiment.py

# Experiment 3: Data augmentation only
python IADAF_experiment.py

# Experiment 4: Physical constraints only
python LDPC_experiment.py

# Experiment 5: Complete framework (all modules)
python Complete_Model_experiment.py
```

Runtime: Approximately 30 minutes per experiment on a standard desktop with GPU.

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

These experiments evaluate performance at 10%, 25%, 50%, 75%, and 100% of training data.

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

These experiments inject Gaussian noise at 5%, 10%, 15%, and 20% levels.

## Repository Structure

```
physics-informed-ml-framework/
│
├── README.md
├── requirements.txt
│
├── data/
│   │
│   ├── solubility/
│   │   ├── raw/
│   │   │   ├── ternary/
│   │   │   │   ├── KCl_MgCl2_H2O.xlsx
│   │   │   │   ├── NaCl_KCl_H2O.xlsx
│   │   │   │   └── NaCl_MgCl2_H2O.xlsx
│   │   │   └── binary/
│   │   │       ├── KCl_H2O.xlsx
│   │   │       ├── MgCl2_H2O.xlsx
│   │   │       └── NaCl_H2O.xlsx
│   │   │
│   │   ├── cleaned/
│   │   │   ├── KCl_MgCl2_H2O_cleaned.xlsx
│   │   │   ├── NaCl_KCl_H2O_cleaned.xlsx
│   │   │   └── NaCl_MgCl2_H2O_cleaned.xlsx
│   │   │
│   │   ├── split_by_temperature/
│   │   │   ├── KCl_MgCl2_H2O_low_temp.xlsx
│   │   │   ├── KCl_MgCl2_H2O_high_temp.xlsx
│   │   │   ├── NaCl_KCl_H2O_low_temp.xlsx
│   │   │   ├── NaCl_KCl_H2O_high_temp.xlsx
│   │   │   ├── NaCl_MgCl2_H2O_low_temp.xlsx
│   │   │   └── NaCl_MgCl2_H2O_high_temp.xlsx
│   │   │
│   │   └── fixed_splits/
│   │       ├── train.xlsx
│   │       ├── val.xlsx
│   │       └── test.xlsx
│   │
│   └── viscosity/
│       ├── raw/
│       │   ├── ternary/
│       │   │   └── MCH_cis_Decalin_HMN.xlsx
│       │   └── binary/
│       │       ├── MCH_cis_Decalin.xlsx
│       │       ├── MCH_HMN.xlsx
│       │       └── cis_Decalin_HMN.xlsx
│       ├── cleaned/
│       │   └── MCH_cis_Decalin_HMN_cleaned.xlsx
│       └── split_by_temperature/
│           ├── low_temp.xlsx
│           └── high_temp.xlsx
│
├── src/
│   ├── __init__.py
│   ├── binary_predictor.py
│   ├── t_kmeans_lof.py
│   ├── iadaf.py
│   ├── ldpc_solubility.py
│   ├── ldpc_viscosity.py
│   ├── solubility_pipeline.py
│   ├── viscosity_pipeline.py
│   ├── utils_solubility.py
│   └── utils_viscosity.py
│
├── models/
│   ├── solubility/
│   │   └── binary/
│   │       ├── KCl_H2O.pth
│   │       ├── MgCl2_H2O.pth
│   │       └── NaCl_H2O.pth
│   │
│   └── viscosity/
│       └── binary/
│           ├── cis_Decalin_HMN.pth
│           ├── MCH_cis_Decalin.pth
│           └── MCH_HMN.pth
│
├── experiments/
│   │
│   ├── solubility/
│   │   ├── __init__.py
│   │   │
│   │   ├── ablation/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_experiment.py
│   │   │   ├── T_KMeans_LOF_experiment.py
│   │   │   ├── IADAF_experiment.py
│   │   │   ├── LDPC_experiment.py
│   │   │   └── Complete_Model_experiment.py
│   │   │
│   │   ├── small_sample/
│   │   │   ├── __init__.py
│   │   │   ├── small_sample_baseline_experiment.py
│   │   │   ├── small_sample_T_KMeans_LOF_experiment.py
│   │   │   ├── small_sample_IADAF_experiment.py
│   │   │   ├── small_sample_LDPC_experiment.py
│   │   │   └── small_sample_Complete_Model_experiment.py
│   │   │
│   │   └── noise/
│   │       ├── __init__.py
│   │       ├── noise_robustness_baseline_experiment.py
│   │       ├── noise_robustness_T_KMeans_LOF_experiment.py
│   │       ├── noise_robustness_IADAF_experiment.py
│   │       ├── noise_robustness_LDPC_experiment.py
│   │       └── noise_robustness_Complete_Model_experiment.py
│   │
│   └── viscosity/
│       ├── __init__.py
│       └── ablation/
│           ├── __init__.py
│           ├── baseline_experiment.py
│           └── Complete_Model_experiment.py
│
└── results/
    ├── solubility/
    │   ├── ablation/
    │   │   ├── baseline_results/
    │   │   ├── T_KMeans_LOF_results/
    │   │   ├── IADAF_results/
    │   │   ├── LDPC_results/
    │   │   └── Complete_Model_results/
    │   │
    │   ├── small_sample/
    │   │   ├── small_sample_baseline_results/
    │   │   ├── small_sample_T_KMeans_LOF_results/
    │   │   ├── small_sample_IADAF_results/
    │   │   ├── small_sample_LDPC_results/
    │   │   └── small_sample_Complete_Model_results/
    │   │
    │   └── noise/
    │       ├── noise_robustness_baseline_results/
    │       ├── noise_robustness_T_KMeans_LOF_results/
    │       ├── noise_robustness_IADAF_results/
    │       ├── noise_robustness_LDPC_results/
    │       └── noise_robustness_Complete_Model_results/
    │
    └── viscosity/
        └── ablation/
            ├── baseline_results/
            └── Complete_Model_results/
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

### Viscosity System

**MCH-cis-Decalin-HMN** (Methylcyclohexane-cis-Decalin-Heptamethylnonane)
- Temperature range: 20-80°C
- Pressure range: 0.1-100 MPa
- Total datapoints: 546
- Training: 20-40°C (136 points)
- Testing: 40-80°C (387 points, high-temperature extrapolation)

### Data Organization

**raw/**: Original experimental data from literature

**cleaned/**: Data after T-KMeans-LOF outlier detection, demonstrating the effect of data quality control

**split_by_temperature/**: Temperature-based train/test splits
- Low-temperature data: Used for k-fold cross-validation during training
- High-temperature data: Reserved for evaluating extrapolation capability

**fixed_splits/**: Fixed train/val/test splits for KCl-MgCl2-H2O system, enabling direct comparison across model configurations

### Data Format

**Solubility data** (Excel format):
- Temperature (°C)
- Component 1 mass fraction (%)
- Component 2 mass fraction (%)
- Solubility (target variable)

**Viscosity data** (Excel format):
- Temperature (°C)
- Pressure (MPa)
- x1 (mole fraction of component 1)
- x2 (mole fraction of component 2)
- Viscosity (mPa·s, target variable)

## Experimental Design

### Ablation Studies

Five experimental configurations evaluate the independent contributions and synergistic effects of each module:

**Experiment 1: Baseline Model**
- Standard DNN trained on raw data
- No outlier detection, no augmentation, no physical constraints
- Establishes baseline performance

**Experiment 2: T-KMeans-LOF**
- Outlier detection applied to raw data
- Standard DNN trained on cleaned data
- Evaluates impact of data quality control

**Experiment 3: IADAF**
- Data augmentation applied to raw data
- Standard DNN trained on augmented data
- Evaluates impact of synthetic data generation

**Experiment 4: LDPC**
- Standard DNN trained on raw data
- Physical constraints applied during inference
- Evaluates impact of boundary consistency enforcement

**Experiment 5: Complete Model**
- Sequential integration of all three modules
- Raw data → T-KMeans-LOF cleaning → IADAF augmentation → DNN training → LDPC constraint correction
- Demonstrates synergistic effects

### Small-Sample Robustness Tests

Evaluates framework performance under data scarcity:

- Temperature-stratified sampling at five gradients: 10%, 25%, 50%, 75%, 100%
- Sample sizes range from 32 to 320 datapoints
- 20 independent sampling realizations per gradient
- Tests data efficiency and degradation patterns

### Noise Robustness Tests

Evaluates framework tolerance to measurement errors:

- Gaussian noise injection at five levels: 0%, 5%, 10%, 15%, 20%
- Noise applied to target variable in training set
- 20 independent noise realizations per level
- Tests outlier detection effectiveness and stability

All experiments use a fixed high-temperature test set (T > 100°C, 135 datapoints) to ensure consistent evaluation conditions.

## Framework Components

### Module 1: T-KMeans-LOF

Temperature-guided outlier detection based on the thermodynamic stability principle.

**Key principle**: Equilibrium and transport properties must vary smoothly with state parameters due to analyticity of microscopic interaction potentials and Cahn-Hilliard energy minimization.

**Methodology**:
- K-means clustering stratifies data into near-isothermal layers
- Local Outlier Factor (LOF) detects density-based anomalies within each cluster
- Robust Moving Median (RMM) identifies deviations from thermodynamic smoothness
- Consensus scoring combines both methods to balance detection sensitivity

**Key parameters**:
- contamination: Expected proportion of outliers (default: 0.2)
- n_clusters: Number of temperature clusters (default: determined by silhouette analysis)
- n_neighbors: LOF neighbors parameter (default: 5)

### Module 2: IADAF

Iterative adaptive data augmentation using WGAN-GP with Bayesian optimization.

**Key principle**: Learn intrinsic distribution of cleaned data to generate synthetic samples, transforming discrete experimental points into a continuous data manifold.

**Methodology**:
- WGAN-GP generates synthetic samples with Wasserstein distance and gradient penalty
- Bayesian optimization automatically tunes hyperparameters for each system
- XGBoost discriminator filters generated samples for quality control
- DNN validation R2 serves as ultimate quality evaluator

**Optimized hyperparameters**:
- latent_dim: Latent space dimensionality (range: 1-100)
- hidden_dim: Hidden layer dimensionality (range: 150-350)
- lambda_gp: Gradient penalty coefficient (range: 0.05-5.0)

### Module 3: LDPC

Low-dimensional physical constraints based on topological continuity of macroscopic material systems.

**Key principle**: Multicomponent systems must degenerate to corresponding low-dimensional subsystems at composition boundaries.

**Methodology**:
- Binary subsystem models predict boundary values at each temperature
- Dynamic correction during inference (not embedded in training loss)
- Exponential decay weighting ensures boundary accuracy while preserving interior predictions
- Boundary constraints propagate to interior points through interpolation

**Solubility systems**: Two binary boundaries (Component 1-Water, Component 2-Water)

**Viscosity systems**: Three binary boundaries (MCH=0, Decalin=0, HMN=0)

**Key parameters**:
- decay_rate: Controls extent of boundary correction (default: k=2)
- Binary models: Ensemble learning with Bootstrap sampling

## Core Module Files

**binary_predictor.py**: Binary boundary prediction models for LDPC physical constraints, supporting ensemble learning and GPU acceleration

**t_kmeans_lof.py**: Temperature-guided outlier detection combining K-means clustering, LOF density analysis, and RMM trend analysis

**iadaf.py**: Adaptive data augmentation with WGAN-GP, Bayesian hyperparameter optimization, and TSTR quality evaluation

**ldpc_solubility.py**: Physical constraints for solubility prediction using conforming correction method with two-boundary enforcement

**ldpc_viscosity.py**: Physical constraints for viscosity prediction using edge sampling correction method with three-boundary enforcement

**solubility_pipeline.py**: End-to-end workflow integrating all modules for solubility systems, supporting 5-fold cross-validation

**viscosity_pipeline.py**: End-to-end workflow integrating all modules for viscosity systems, supporting PARL-DNN architecture

**utils_solubility.py**: Physics evaluation for solubility, including boundary consistency (2 boundaries) and thermodynamic smoothness (2D Laplacian)

**utils_viscosity.py**: Physics evaluation for viscosity, including boundary consistency (3 boundaries) and thermodynamic smoothness (4D Laplacian)

## Performance Metrics

### Statistical Metrics

- **R2 (Coefficient of Determination)**: Model fit quality
- **RMSE (Root Mean Square Error)**: Prediction accuracy
- **MAE (Mean Absolute Error)**: Average deviation

### Physics-Based Metrics

**Physical Rationality**: Overall compliance with thermodynamic laws
- Calculated as mean of Boundary Consistency and Thermodynamic Smoothness

**Boundary Consistency**: Convergence accuracy to binary subsystem boundaries
- Measured using normalized RMSE at compositional limits
- Evaluates system degeneracy at w(Component) = 0

**Thermodynamic Smoothness**: Surface continuity based on Cahn-Hilliard theory
- Quantifies extreme Laplacian values using P99 quantile method
- Penalizes non-physical oscillations violating energy minimization

## Key Dependencies

Core packages with exact versions specified in requirements.txt:

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
- **Recommended**: NVIDIA GPU with CUDA support and 16GB RAM
- **Storage**: Approximately 2 GB for code, data, and results

### Runtime

Approximate execution times on Intel i7 CPU with NVIDIA RTX 3080 GPU:

- Single ablation experiment: 30 minutes
- All solubility ablation studies (5 configurations): 2.5 hours
- Small-sample robustness tests: 2 hours
- Noise robustness tests: 2 hours
- Complete experimental suite: 8-10 hours

## Results Organization

Each experiment automatically saves results to the results/ directory:

**Per experiment output**:
- metrics.csv: Performance metrics (R2, MAE, RMSE, physics scores)
- predictions.csv: Model predictions on test set
- config.json: Experimental configuration
- training_history.csv: Loss curves during training
- physics_scores.csv: Physical consistency evaluations

**Example structure**:
```
results/solubility/ablation/Complete_Model_results/
├── metrics.csv
├── predictions.csv
├── config.json
├── training_history.csv
└── physics_scores.csv
```

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory

**Solution**: Reduce batch size or use CPU mode:
```python
pipeline = SolubilityPipeline(device='cpu')
```

**Issue**: Excel file reading error

**Solution**: Install or upgrade openpyxl:
```bash
pip install openpyxl --upgrade
```

**Issue**: Results differ from paper

**Solution**: Ensure exact dependency versions:
```bash
pip install -r requirements.txt --force-reinstall
```

**Issue**: Long training time

**Solution**: Reduce epochs for testing or enable early stopping:
```python
pipeline = SolubilityPipeline(
    max_epochs=500,
    early_stopping=True
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2024physics,
  title={Physics-Constrained Multi-Module Machine Learning Framework for Scientific Prediction from Small-Sample Experimental Data},
  author={Wang, Yuan and Wang, Tiancheng and Du, Jindi and Chen, Qian and Zhang, Weidong and Liu, Dahuan},
  journal={Nature Communications},
  year={2024},
  note={Under review}
}
```

## Code Availability

The source code is permanently archived on Zenodo: https://doi.org/10.5281/zenodo.15624558

For the latest updates, visit: https://github.com/wang201111/hybrid-intelligence-framework

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
