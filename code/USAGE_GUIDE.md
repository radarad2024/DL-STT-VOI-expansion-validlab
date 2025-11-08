# Training and Validation Scripts Usage Guide

This directory contains two separate scripts for training and external validation with proper statistical methods.

## Overview

### 1. `train_5cv.py` - 5-Fold Cross-Validation Training
- **Purpose**: Train models with 5-fold cross-validation
- **Statistics**: Bootstrap confidence intervals (2000 samples)
- **Threshold**: Youden's Index optimized on training data
- **Output**: Optimal thresholds for each model (mean across 5 folds)

### 2. `evaluate_external.py` - External Validation
- **Purpose**: Evaluate on external test set
- **Statistics**: DeLong method for AUC CI and model comparisons
- **Threshold**: Uses training-derived optimal thresholds
- **Output**: Performance metrics and statistical comparisons

---

## Script 1: Training (5-Fold CV)

### Configuration

Edit the `Config` class in `train_5cv.py`:

```python
class Config:
    # Paths (MODIFY THESE)
    root_dir = "/path/to/Soft Tissue Tumor VOI"
    label_file = "labels.csv"
    output_dir = "./outputs/5cv_results"

    # Model configurations
    model_configs = {
        "Rectangular VOI": "normalization_BoundingBox",
        "StandardVOI": "normalization_ROI",
        "ExpandedVOI": "normalization_Dilation1cm"
    }

    # Training settings
    n_folds = 5
    epochs = 30
    batch_size = 32
    learning_rate = 1e-4
```

### Running Training

```bash
python train_5cv.py
```

### Expected Output Structure

```
outputs/5cv_results/
├── Rectangular_VOI/
│   ├── fold_1/
│   │   ├── results.json
│   │   ├── test_predictions.csv
│   │   └── roc_curve.png
│   ├── fold_2/ ... fold_5/
│   └── aggregated_results.json
├── StandardVOI/
│   └── [same structure]
├── ExpandedVOI/
│   └── [same structure]
├── optimal_thresholds.json          ← IMPORTANT: Used for external validation
├── model_comparison.csv
└── model_comparison.md
```

### Key Output Files

#### `optimal_thresholds.json`
Contains mean optimal thresholds across 5 folds:
```json
{
  "Rectangular VOI": 0.4523,
  "StandardVOI": 0.5012,
  "ExpandedVOI": 0.4891
}
```

#### Each fold's `results.json`
```json
{
  "fold": 1,
  "train_threshold": 0.4523,
  "test_auc": 0.856,
  "test_ci_lower": 0.812,
  "test_ci_upper": 0.899,
  "test_metrics": {
    "sensitivity": 0.823,
    "specificity": 0.791,
    ...
  }
}
```

---

## Script 2: External Validation

### Prerequisites

1. **Trained models** and saved predictions on external test set
2. **optimal_thresholds.json** from training script

### Prepare External Predictions CSV

Your CSV should have the following structure:

```csv
True_Label,BoundingBox_Prob,StandardVOI_Prob,ExpandedVOI_Prob
1,0.8234,0.7891,0.8456
0,0.2341,0.3123,0.2987
1,0.9012,0.8567,0.9234
...
```

### Configuration

Edit the `Config` class in `evaluate_external.py`:

```python
class Config:
    # Paths (MODIFY THESE)
    predictions_csv = "/path/to/external_predictions.csv"
    optimal_thresholds_json = "./outputs/5cv_results/optimal_thresholds.json"
    output_dir = "./outputs/external_validation"

    # Column names
    label_column = 'True_Label'
    model_prob_columns = {
        'Rectangular VOI': 'BoundingBox_Prob',
        'StandardVOI': 'StandardVOI_Prob',
        'ExpandedVOI': 'ExpandedVOI_Prob'
    }
```

### Running External Validation

```bash
python evaluate_external.py
```

### Expected Output Structure

```
outputs/external_validation/
├── external_validation_results.json    # Complete results with all metrics
├── external_validation_summary.csv     # Summary table
├── external_validation_summary.md      # Markdown report
├── roc_curves.png                      # ROC curves (PNG)
└── roc_curves.pdf                      # ROC curves (PDF)
```

### Key Output Files

#### `external_validation_results.json`
```json
{
  "n_samples": 150,
  "n_positive": 78,
  "n_negative": 72,
  "models": {
    "Rectangular VOI": {
      "auc": 0.867,
      "ci_lower": 0.823,
      "ci_upper": 0.911,
      "ci_method": "DeLong",
      "training_threshold": 0.4523,
      "metrics": {
        "sensitivity": 0.846,
        "specificity": 0.792,
        ...
      }
    },
    ...
  },
  "delong_tests": {
    "Rectangular VOI vs StandardVOI": {
      "auc_diff": 0.0234,
      "z_statistic": 2.1456,
      "p_value": 0.0319,
      "significant": true
    },
    ...
  }
}
```

#### `external_validation_summary.md`
Readable markdown report with:
- Performance metrics table
- DeLong test results
- Notes on methodology

---

## Statistical Methods Explained

### Training (5-Fold CV)

#### Bootstrap Confidence Intervals
- **Method**: Percentile bootstrap with 2000 resamples
- **Purpose**: Estimate uncertainty in AUC
- **Application**: Each fold's test set
- **Formula**: 95% CI = [2.5th percentile, 97.5th percentile]

#### Youden's Index
- **Method**: Maximize (Sensitivity + Specificity - 1)
- **Purpose**: Find optimal threshold
- **Application**: Training data only
- **Use**: Applied to test data (including external)

### External Validation

#### DeLong Method
- **AUC Variance**: Based on Mann-Whitney U-statistic theory
- **CI Calculation**: Normal approximation with calculated variance
- **Advantages**:
  - No resampling required
  - Theoretically grounded
  - Efficient computation

#### DeLong Test
- **Purpose**: Compare two correlated ROC curves
- **Test Statistic**: Z = (AUC₁ - AUC₂) / SE(AUC₁ - AUC₂)
- **P-value**: Two-tailed test
- **Interpretation**: p < 0.05 indicates significant difference

---

## Complete Workflow Example

### Step 1: Training
```bash
# Configure paths in train_5cv.py
python train_5cv.py
```

**Output**: `outputs/5cv_results/optimal_thresholds.json`

### Step 2: Generate External Predictions

Use your trained models to generate predictions on external test set:

```python
# Pseudo-code
for model in models:
    predictions = model.predict(external_data)
    save_predictions(predictions)
```

Create CSV with required columns.

### Step 3: External Validation
```bash
# Configure paths in evaluate_external.py
python evaluate_external.py
```

**Output**: Complete evaluation with DeLong statistics

---

## Key Differences Between Scripts

| Aspect | Training (5-CV) | External Validation |
|--------|----------------|---------------------|
| **Dataset** | Internal (5-fold split) | External (completely separate) |
| **AUC CI Method** | Bootstrap (2000 samples) | DeLong |
| **Threshold Source** | Optimized on training folds | Loaded from training |
| **Model Comparison** | Not applicable* | DeLong test |
| **Purpose** | Model development | Model validation |

*Note: Comparing models across different CV folds requires different statistical methods not implemented here.

---

## Interpreting Results

### Training Results

Look at `model_comparison.md`:
- **Mean AUC ± SD**: Performance across 5 folds
- **Threshold**: Optimal cutoff for external use
- **Sensitivity/Specificity**: Performance at optimal threshold

### External Validation Results

Look at `external_validation_summary.md`:

1. **AUC with DeLong CI**: How well the model generalizes
2. **Performance at Training Threshold**: Real-world performance
3. **DeLong Test P-values**: Whether models are significantly different

**Example Interpretation**:
```
Rectangular VOI vs StandardVOI: p = 0.032
→ Significant difference exists between models

StandardVOI vs ExpandedVOI: p = 0.156
→ No significant difference
```

---

## Troubleshooting

### Common Issues

#### 1. "File not found" errors
- Check that `root_dir` points to correct data directory
- Verify `optimal_thresholds.json` exists before running external validation

#### 2. "Column not found" in external validation
- Ensure CSV column names match `model_prob_columns` in Config

#### 3. GPU out of memory
- Reduce `batch_size` in Config
- Use fewer `num_workers`

#### 4. Statistical errors (division by zero)
- Ensure test set has both classes (positive and negative samples)
- Check for NaN values in predictions

---

## Citation

If you use these scripts, please ensure proper statistical reporting:

**For Training**:
> "Model performance was evaluated using 5-fold cross-validation with stratified sampling. AUC confidence intervals were calculated using bootstrap method with 2000 resamples. Optimal classification thresholds were determined using Youden's Index on training data."

**For External Validation**:
> "External validation was performed on an independent test set (n=X). AUC confidence intervals were calculated using the DeLong method. Model comparisons were performed using DeLong test for correlated ROC curves. Classification thresholds were applied from training data without re-optimization."

---

## References

1. DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 837-845.

2. Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32-35.

3. Efron, B., & Tibshirani, R. J. (1994). *An introduction to the bootstrap*. CRC press.

---

## Contact & Support

For issues or questions:
1. Check configuration settings
2. Verify data format
3. Review error messages carefully
4. Ensure all dependencies are installed

**Required packages**:
```
numpy
pandas
scikit-learn
scipy
matplotlib
torch
monai
```
