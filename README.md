# DL-STT-VOI-Expansion

Deep learning-based classification of soft tissue tumors on MRI: evaluating the impact of peritumoral VOI expansion.

## Overview

This project develops and validates DenseNet-121 models for differentiating benign and malignant soft tissue tumors using three voxel-of-interest (VOI) strategies on multi-sequence MRI (T1WI, T2WI, contrast-enhanced T1WI):

| VOI | Description |
|-----|-------------|
| **R-VOI** | Rectangular bounding box tightly enclosing the tumor |
| **S-VOI** | Standard tumor segmentation along tumor margins |
| **E-VOI** | Expanded VOI with 1 cm peritumoral extension |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | DenseNet-121 (MONAI) |
| Input | 3-channel (T1WI + T2WI + CE-T1WI), 128×128×80 voxels |
| Voxel spacing | 1×1×2 mm³ |
| Optimizer | AdamW (weight_decay=1e-5) |
| Learning rate | 5×10⁻⁵ |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) |
| Batch size | 32 |
| Epochs | 30 |
| Loss | Cross-entropy with inverse class-frequency weights |
| Mixed precision | Enabled |
| Normalization | f(x) = (x − μ) / σ × 1000 (nonzero voxels) |

## Notebooks

| File | Description |
|------|-------------|
| `01_internal_validation.ipynb` | 5-fold CV training, threshold determination, ROC curves |
| `02_external_validation.ipynb` | External cohort evaluation, DeLong tests, ROC curves |

### Internal validation (`01`)
- Stratified 5-fold cross-validation
- Trains all three VOI models and saves checkpoints
- Trains final models on full training data
- Determines classification threshold per model (sensitivity ≥ 80%, maximum specificity)
- Generates ROC curves with ±1 SD shading

### External validation (`02`)
- Loads prediction results (CSV) and applies predetermined thresholds
- AUC with DeLong 95% CI
- Sensitivity and specificity with exact binomial (Clopper-Pearson) 95% CI
- Pairwise DeLong tests (models vs. models, models vs. readers, reader vs. reader)
- Generates ROC curves with AUC and 95% CI

## Data Structure

```
data_dir/
├── EN/                    # Contrast-enhanced T1WI
│   └── patient_id/
│       ├── image.nii.gz
│       └── seg_label.nii.gz
├── T1/                    # T1WI
│   └── patient_id/
│       └── image.nii.gz
└── T2/                    # T2WI
    └── patient_id/
        └── image.nii.gz
```

**labels.csv**: `patient_id, label` (0 = benign, 1 = malignant)

## Requirements

```
torch >= 1.12
monai >= 1.0
scikit-learn
scipy
pandas
numpy
matplotlib
```

## License

This project is for academic research purposes.
