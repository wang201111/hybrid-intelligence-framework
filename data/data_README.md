# Data Documentation

## Overview

This directory contains experimental data for validating the physics-constrained machine learning framework across four chemical systems. The data organization is carefully designed to support rigorous evaluation of model generalization and high-temperature extrapolation capabilities.

## Critical Design Principle: Temperature-Based Data Management

The framework's key innovation is predicting high-temperature properties by training only on low-temperature data. This requires careful data management to ensure fair evaluation:

**High-Temperature Data (Test Set):**
- Pre-cleaned using T-KMeans-LOF to serve as gold-standard reference
- Used ONLY for testing, never for training
- Ensures fair evaluation across all model configurations

**Low-Temperature Data (Training Set):**
- Retained with experimental noise in raw form
- Different models process this data differently (see Model-Specific Processing below)
- This design tests each model's ability to handle real-world noisy data

## Data Sources

All data are compiled from publicly available literature:

**Solubility Systems:**
- KCl-MgCl2-H2O: References 57-63 from manuscript
  - Mallet (1902), Saunders (1899), Schrelnemakers (1909), etc.
- NaCl-KCl-H2O: References 57-60, 62
- NaCl-MgCl2-H2O: References 60-61, 64-65
  - Yang et al. (2013), Charykova & Charykov (2007), etc.

**Viscosity System:**
- MCH-cis-Decalin-HMN: Reference 69
  - Zéberg-Mikkelsen et al. (2003). J. Chem. Eng. Data, 48, 1387-1392

## Data Format Specifications

### Solubility Data Format
```
Column 1: Temperature (°C)
Column 2: Mass fraction of component 1 (%, e.g., w(KCl))
Column 3: Mass fraction of component 2 (%, e.g., w(MgCl2))
```

Example (KCl-MgCl2-H2O):
```
Temperature | w(KCl)/% | w(MgCl2)/%
-34.5       | 0.0      | 57.51
-10.0       | 5.2      | 48.3
25.0        | 15.8     | 35.6
```

### Viscosity Data Format
```
Column 1: Temperature (°C)
Column 2: Pressure (MPa)
Column 3: Mole fraction of MCH (x1)
Column 4: Mole fraction of cis-Decalin (x2)
Column 5: Viscosity (mPa·s)
```

Example:
```
Temperature | Pressure | x(MCH) | x(cis-Dec) | Viscosity
20.0        | 0.1      | 0.333  | 0.333      | 1.234
40.0        | 10.0     | 0.500  | 0.250      | 1.456
```

## Directory Structure

```
data/
├── solubility/
│   ├── raw/
│   │   ├── ternary/              # Original three-component data
│   │   │   ├── KCl_MgCl2_H2O.xlsx       (619 points, -34°C to 227°C)
│   │   │   ├── NaCl_KCl_H2O.xlsx        (463 points, -23°C to 180°C)
│   │   │   └── NaCl_MgCl2_H2O.xlsx      (313 points, -35°C to 200°C)
│   │   └── binary/               # Two-component boundary data (for LDPC)
│   │       ├── KCl_H2O.xlsx
│   │       ├── MgCl2_H2O.xlsx
│   │       └── NaCl_H2O.xlsx
│   │
│   ├── cleaned/                  # After T-KMeans-LOF processing
│   │   └── [System]_cleaned.xlsx
│   │
│   ├── split_by_temperature/     # Temperature-based train/test splits
│   │   ├── KCl_MgCl2_H2O_low_temp.xlsx   # For 5-fold CV (484 points, -34~100°C)
│   │   ├── KCl_MgCl2_H2O_high_temp.xlsx  # For testing (135 points, 100~227°C)
│   │   └── ... (similarly for other systems)
│   │
│   └── fixed_splits/             # Fixed splits for specific experiments
│       ├── train.xlsx            # Low-temp training set (raw, with noise)
│       ├── val.xlsx              # Low-temp validation set (cleaned)
│       └── test.xlsx             # High-temp test set (cleaned)
│
└── viscosity/
    └── [Similar structure as solubility]
```

## Temperature-Based Splitting Strategy

All systems use temperature-based splitting to evaluate high-temperature extrapolation:

### KCl-MgCl2-H2O System
- **Total datapoints**: 619 (from -34.5°C to 227°C)
- **Low-temperature (training)**: 484 points from -34.5°C to 100°C
  - Used for model training and validation
  - Retained with experimental noise in raw form
- **High-temperature (testing)**: 135 points from 100°C to 227°C
  - Pre-cleaned with T-KMeans-LOF (contamination=0.20)
  - Reserved as gold-standard test set
  - Evaluates extrapolation to unseen temperature range

### NaCl-KCl-H2O System
- **Total datapoints**: 463 (from -23°C to 180°C)
- **Low-temperature**: 316 points from -23°C to 50°C
- **High-temperature**: 147 points from 50°C to 180°C

### NaCl-MgCl2-H2O System
- **Total datapoints**: 313 (from -35°C to 200°C)
- **Low-temperature**: 244 points from -35°C to 100°C
- **High-temperature**: 69 points from 100°C to 200°C

### MCH-cis-Decalin-HMN System
- **Total datapoints**: 546 (20°C to 80°C, 0.1 to 100 MPa)
- **Low-temperature**: 136 points from 20°C to 40°C
- **High-temperature**: 387 points from 40°C to 80°C

## Experimental Design and Data Processing Protocols

### Design Philosophy

The framework's effectiveness depends on how different modules handle noisy, small-sample data. To ensure fair evaluation, we designed a rigorous data management protocol:

1. **Test Set (High-Temperature)**: Pre-cleaned to provide consistent gold-standard reference
2. **Training Set (Low-Temperature)**: Kept in raw form to test each model's noise-handling capability
3. **Model-Specific Processing**: Different models apply different preprocessing strategies

### Model-Specific Data Processing

#### Baseline Model (Pure DNN)
- **Training data**: Uses raw low-temperature data (with noise)
- **Rationale**: Represents standard deep learning approach without any preprocessing
- **Challenge**: Must learn from noisy, sparse data directly

#### T-KMeans-LOF Model
- **Training data**: Applies T-KMeans-LOF cleaning to low-temperature data FIRST
- **Processing**: Removes ~15-20% outliers (contamination=0.20)
- **Rationale**: Tests if data cleaning alone improves prediction
- **Challenge**: Smaller dataset after cleaning may reduce model capacity

#### IADAF Model
- **Training data**: Uses raw low-temperature data
- **Processing**: Generates 1500 synthetic samples via WGAN-GP
- **Rationale**: Tests if data augmentation alone helps with extrapolation
- **Challenge**: Synthetic data quality depends on learning from noisy originals

#### LDPC Model
- **Training data**: Uses raw low-temperature data
- **Processing**: No preprocessing, applies constraints during inference only
- **Rationale**: Tests if physical constraints alone ensure consistency
- **Challenge**: Cannot compensate for poor training data quality

#### Complete Model (Proposed Framework)
- **Step 1**: Apply T-KMeans-LOF to low-temperature data → cleaned data
- **Step 2**: Apply IADAF to cleaned data → augmented dataset
- **Step 3**: Train DNN on augmented dataset
- **Step 4**: Apply LDPC constraints during inference
- **Rationale**: Synergistic integration of all three modules
- **Advantage**: Addresses data quality, quantity, and physical consistency simultaneously

### Cross-Validation Strategy

#### 5-Fold Cross-Validation (Ablation Experiments)
```
Low-temperature data (e.g., 484 points) → Split into 5 folds
For each fold:
    - 80% (387 points) → Training set
    - 20% (97 points) → Validation set
    - Model-specific preprocessing applied to training set
    - Validation set evaluated on both statistical and physics metrics
High-temperature data (135 points) → Fixed test set (pre-cleaned)
```

**Implementation Details:**
- Stratified sampling ensures temperature distribution consistency
- Each fold tests the same model configuration
- Results averaged across 5 folds with standard deviation reported
- Test set remains constant across all folds

#### Fixed Split (Small-Sample & Noise Robustness Experiments)

For experiments requiring controlled data perturbation (sample size variation or noise injection), we use fixed splits:

**File Structure:**
```
fixed_splits/
├── train.xlsx        # Low-temperature training set (raw, with noise)
├── val.xlsx          # Low-temperature validation set (cleaned)
└── test.xlsx         # High-temperature test set (cleaned)
```

**Why Different Processing for train.xlsx vs. val.xlsx?**

This asymmetric design serves specific experimental purposes:

1. **train.xlsx (raw, with noise)**:
   - Allows controlled manipulation for noise robustness tests
   - Can add Gaussian noise at 5%, 10%, 15%, 20% levels
   - Can subsample at 10%, 25%, 50%, 75%, 100% levels
   - Tests model's ability to handle degraded training conditions

2. **val.xlsx (cleaned)**:
   - Provides stable reference for hyperparameter selection
   - Prevents validation performance from fluctuating due to noise
   - Ensures consistent comparison across different training conditions

3. **test.xlsx (cleaned, high-temperature)**:
   - Gold-standard reference for final evaluation
   - Consistent across ALL experiments
   - Measures true extrapolation capability

**Example: Small-Sample Experiment**
```python
# Load fixed splits
train_raw = load('fixed_splits/train.xlsx')  # 320 points (low-temp, raw)
val_clean = load('fixed_splits/val.xlsx')    # 80 points (low-temp, cleaned)
test_clean = load('fixed_splits/test.xlsx')  # 135 points (high-temp, cleaned)

# Subsample training set at different ratios
for ratio in [0.10, 0.25, 0.50, 0.75, 1.00]:
    train_subset = sample(train_raw, ratio)  # e.g., 32, 80, 160, 240, 320 points
    
    # Model-specific processing
    if model == 'T-KMeans-LOF':
        train_processed = clean_outliers(train_subset)
    elif model == 'IADAF':
        train_processed = augment_data(train_subset)
    elif model == 'Complete':
        train_processed = clean_outliers(train_subset)
        train_processed = augment_data(train_processed)
    else:  # Baseline or LDPC
        train_processed = train_subset  # No preprocessing
    
    # Train and evaluate
    model.fit(train_processed)
    val_score = model.evaluate(val_clean)
    test_score = model.evaluate(test_clean)
```

**Example: Noise Robustness Experiment**
```python
# Load fixed splits
train_raw = load('fixed_splits/train.xlsx')
val_clean = load('fixed_splits/val.xlsx')
test_clean = load('fixed_splits/test.xlsx')

# Inject Gaussian noise into training target variable
for noise_level in [0.05, 0.10, 0.15, 0.20]:
    train_noisy = add_gaussian_noise(train_raw, noise_level)
    
    # Model-specific processing
    if model == 'T-KMeans-LOF':
        train_processed = clean_outliers(train_noisy)  # Should remove added noise
    # ... (similar to above)
    
    # Evaluate noise tolerance
    model.fit(train_processed)
    test_score = model.evaluate(test_clean)
```

## Data Quality and Outlier Characteristics

### Raw Data Quality Issues

Multi-source experimental data compilation introduces systematic inconsistencies:

1. **Measurement Variability**: Different laboratories, different instruments
2. **Phase Boundary Uncertainties**: Metastable states near saturation
3. **Temperature Control Errors**: Especially at extreme temperatures
4. **Temporal Data Drift**: Experiments conducted across decades (1899-2024)

### T-KMeans-LOF Outlier Detection Parameters

**Default Configuration:**
```python
contamination = 0.20      # Expect ~20% outliers in multi-source data
k_neighbors = 5          # LOF parameter (requires 5-10× sample size)
window_size = 5          # RMM sliding window for trend analysis
n_clusters = 'auto'      # Determined by silhouette score analysis
```

**Why contamination=0.20?**
- Empirically validated on historical multi-source salt-water data
- Conservative estimate accounting for:
  - Inter-laboratory differences: ~10%
  - Metastable phase data: ~5%
  - Measurement errors: ~5%
- See Supplementary Note 2 for sensitivity analysis

**Outlier Removal Statistics:**
```
KCl-MgCl2-H2O low-temp:  484 → 387 points (20.0% removed)
NaCl-KCl-H2O low-temp:   316 → 253 points (19.9% removed)
NaCl-MgCl2-H2O low-temp: 244 → 195 points (20.1% removed)
```

High-temperature test sets: Pre-cleaned with same parameters

## Data Augmentation with IADAF

After outlier removal, cleaned datasets are augmented with synthetic samples:

**Augmentation Configuration:**
```python
n_synthetic = 1500                # Samples to generate per system
latent_dim_range = (1, 100)       # Bayesian optimization search space
hidden_dim_range = (150, 350)     # Bayesian optimization search space
n_iterations = 100                # Bayesian optimization budget
```

**Quality Control:**
- XGBoost discriminator filters low-quality samples (threshold: 15 wt%)
- Validation R² on real data used as optimization objective
- Final synthetic dataset: 1500 high-quality samples per system

**Augmented Dataset Composition:**
```
KCl-MgCl2-H2O:  387 (real) + 1500 (synthetic) = 1887 total
NaCl-KCl-H2O:   253 (real) + 1500 (synthetic) = 1753 total
NaCl-MgCl2-H2O: 195 (real) + 1500 (synthetic) = 1695 total
```

## Critical Note: Real vs. Synthetic Data in Evaluation

**Training:** Uses both real (cleaned) and synthetic (IADAF-generated) data

**Validation:** Uses real data only
- For 5-fold CV: Real data from low-temperature region
- For fixed splits: Pre-cleaned real data (val.xlsx)

**Testing:** Uses ONLY real experimental data from high-temperature region
- Ensures fair evaluation of extrapolation capability
- Prevents synthetic data from inflating test performance
- Provides meaningful comparison with literature results

This protocol ensures that reported test R² values reflect genuine predictive power on real experimental data, not memorization of synthetic patterns.

## Binary System Data (for LDPC Physical Constraints)

Binary subsystem data provide thermodynamic boundary constraints:

### Solubility Binary Systems
```
KCl-H2O:    Saturation curve from 0°C to 200°C
MgCl2-H2O:  Saturation curve from -34°C to 227°C
NaCl-H2O:   Saturation curve from -23°C to 200°C
```

**Usage:**
- Train ensemble DNN models on binary data
- Predict boundary values at any temperature
- Apply exponential decay-weighted corrections to ternary predictions

### Viscosity Binary Systems
```
MCH-cis-Decalin:      Viscosity surface (T, P, x1)
MCH-HMN:              Viscosity surface (T, P, x2)
cis-Decalin-HMN:      Viscosity surface (T, P, x3)
```

**Three-Boundary Constraint:**
- When x(MCH) → 0: Predict using cis-Decalin-HMN boundary
- When x(cis-Dec) → 0: Predict using MCH-HMN boundary  
- When x(HMN) → 0: Predict using MCH-cis-Decalin boundary

## Data Archiving and Reproducibility

**Primary Repository:**
- GitHub: https://github.com/wang201111/hybrid-intelligence-framework/tree/main/data

**Permanent Archive:**
- Zenodo: https://doi.org/10.5281/zenodo.15624559
- Includes all raw data, processed data, and binary system data
- Ensures long-term reproducibility independent of GitHub

**Data Integrity Verification:**
```bash
# MD5 checksums provided in data/checksums.txt
md5sum -c checksums.txt
```

## License and Usage

**Original Data Sources:**
- Retain original publication licenses
- Cited appropriately in manuscript

**Processed Datasets:**
- Shared under Creative Commons Attribution 4.0 (CC-BY-4.0)
- Free to use with proper attribution

**Citation:**
If you use this data, please cite both:
1. The original literature sources (References 57-69 in manuscript)
2. This framework paper: Wang et al. (2025), Nature Communications

## Frequently Asked Questions

**Q: Why are high-temperature data pre-cleaned but not low-temperature data?**

A: This design serves two purposes:
1. **Fair Evaluation**: High-temperature test set must be consistent across all models
2. **Realistic Training**: Low-temperature data retained with noise to test each model's preprocessing capability

**Q: Why is val.xlsx cleaned but train.xlsx is not?**

A: For small-sample and noise robustness experiments:
- train.xlsx (raw) allows controlled noise/subsample manipulation
- val.xlsx (cleaned) provides stable reference for hyperparameter tuning
- Asymmetric design enables rigorous testing of noise tolerance

**Q: Can I use the cleaned data for training?**

A: Yes, but your results won't match the "Baseline" configuration in the paper. Different models use different preprocessing:
- Baseline: Uses raw training data
- T-KMeans-LOF: Uses cleaned training data
- See "Model-Specific Processing" section above

**Q: What if my data has <20% outliers?**

A: Adjust the contamination parameter:
```python
detector = TKMeansLOF(contamination=0.10)  # For high-quality single-source data
detector = TKMeansLOF(contamination=0.30)  # For very noisy multi-source data
```

**Q: Why not clean the test set in-place during evaluation?**

A: Pre-cleaning the test set ensures:
- Consistent gold-standard reference across all experiments
- No accidental information leakage from test to training
- Reproducible results for other researchers

## Contact

For data-related questions:
- Dahuan Liu: liudh@mail.buct.edu.cn
- Weidong Zhang: weidzhang1208@126.com
- GitHub Issues: https://github.com/wang201111/hybrid-intelligence-framework/issues

## References

Complete data source references available in manuscript References 57-69.

Key references:
- Mallet, J.W. (1902). On the Relation of Conductivity. American Chemical Journal, 27, 55-58.
- Zéberg-Mikkelsen et al. (2003). High-pressure viscosity measurements. J. Chem. Eng. Data, 48, 1387-1392.
- See manuscript for complete bibliography.

---

**Last Updated:** January 2025  
**Version:** 1.0  
**Status:** Peer review at Nature Communications
