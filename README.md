# 3D MRI Multi-Sequence Soft Tissue Tumor Classification

## Project Overview

This repository contains the implementation of a deep learning framework for automated classification of soft tissue tumors using multi-parametric 3D MRI data. The framework employs a multi-model ensemble approach with different Volume of Interest (VOI) extraction strategies to improve classification robustness.

### Key Features
- Multi-sequence MRI fusion (T1-weighted, T2-weighted, Contrast-enhanced)
- Three distinct VOI extraction strategies
- 5-fold cross-validation with stratification
- Comprehensive statistical evaluation with bootstrap confidence intervals
- GPU-optimized training with DataParallel support

## System Requirements

### Hardware Requirements
- **Minimum GPU Memory**: 16GB VRAM (single GPU)
- **Recommended Setup**: 2× NVIDIA RTX 4080 (16GB each) or equivalent
- **CPU**: Intel Core i9-14900K or equivalent (≥16 cores recommended)
- **RAM**: 64GB DDR5
- **Storage**: 500GB SSD for dataset and outputs

### Software Dependencies
```
Python 3.9.16
PyTorch 2.0.1
CUDA 11.8
cuDNN 8.7.0
MONAI 1.2.0
```

### Python Package Requirements
```python
torch==2.0.1+cu118
monai==1.2.0
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
scipy==1.10.1
matplotlib==3.7.1
seaborn==0.12.2
tqdm==4.65.0
```

## Dataset Structure

### Directory Organization
```
Soft Tissue Tumor VOI/
│
├── labels.csv                           # Patient labels (patient_id, label)
│
├── normalization_BoundingBox/           # Model 1: Tight bounding box
│   ├── EN/                              # Contrast-enhanced sequences
│   │   └── {patient_id}/
│   │       ├── {patient_id}.nii.gz     # Image volume
│   │       └── seg_{patient_id}.nii.gz # Segmentation mask
│   ├── T1/                              # T1-weighted sequences
│   │   └── {patient_id}/
│   │       └── {patient_id}.nii.gz
│   └── T2/                              # T2-weighted sequences
│       └── {patient_id}/
│           └── {patient_id}.nii.gz
│
├── normalization_ROI/                   # Model 2: Standard VOI
│   └── [Same structure as above]
│
└── normalization_Dilation1cm/           # Model 3: Expanded VOI (+1cm)
    └── [Same structure as above]
```

### Data Specifications
- **Image Format**: NIfTI (.nii.gz)
- **Sequences**: T1-weighted, T2-weighted, Contrast-enhanced (EN)
- **Segmentation**: Binary tumor masks for VOI extraction
- **Labels**: Binary classification (0: benign, 1: malignant)

## Preprocessing Pipeline

### 1. Image Loading
- **Framework**: MONAI LoadImaged
- **Channel Management**: EnsureChannelFirstd
- **Format**: NIfTI with automatic orientation correction

### 2. Spatial Resampling
- **Method**: Trilinear interpolation for images, nearest-neighbor for masks
- **Target Spacing**: [1.0, 1.0, 2.0] mm (x, y, z)
- **Implementation**: MONAI Spacingd

### 3. Volume Standardization
- **Target Size**: [128, 128, 80] voxels
- **Method**: Trilinear resampling for images
- **Implementation**: MONAI Resized

### 4. Intensity Normalization
- **Step 1**: Scale intensity to [0, 1] range (ScaleIntensityd)
- **Step 2**: Mask-based intensity extraction (MaskIntensityd)
- **Step 3**: Z-score normalization within non-zero regions (NormalizeIntensityd)

### 5. Data Augmentation (Training Only)
| Augmentation | Probability | Parameters |
|--------------|------------|------------|
| Random Flip (X-axis) | 0.5 | spatial_axis=0 |
| Random Flip (Y-axis) | 0.5 | spatial_axis=1 |
| Random Affine | 0.8 | rotation=±0.1 rad, translation=±10mm (x,y), ±5mm (z), scale=±10% |
| Gaussian Noise | 0.3 | mean=0, std=0.1 |
| Bias Field | 0.2 | degree=3, coeff_range=(0.0, 0.1) |
| 3D Elastic Deformation | 0.1 | sigma=(5,8), magnitude=(100,200) |

## Model Architecture

### Base Network
- **Architecture**: DenseNet-121 (3D variant)
- **Implementation**: MONAI networks
- **Input Channels**: 3 (concatenated T1, T2, EN)
- **Output Classes**: 2 (binary classification)
- **Spatial Dimensions**: 3D

### Multi-Model Ensemble
1. **Model 1 - BoundingBox**: Tight tumor bounding box extraction
2. **Model 2 - StandardVOI**: Standard VOI with original segmentation
3. **Model 3 - ExpandedVOI**: Dilated VOI with 1cm margin

### Technical Details
- **Total Parameters**: ~7.2M
- **Trainable Parameters**: ~7.2M
- **Growth Rate**: 32
- **Dense Blocks**: 4
- **Compression Factor**: 0.5

## Training Configuration

### Optimization Settings
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | AdamW | Better generalization with weight decay |
| Learning Rate | 1e-4 | Optimal for batch size 32 |
| Weight Decay | 1e-5 | L2 regularization |
| Batch Size | 32 | 16 per GPU with DataParallel |
| Epochs | 30 | Fixed training without early stopping |
| Gradient Clipping | 1.0 | Prevent gradient explosion |
| Mixed Precision | FP16 | Memory efficiency with AMP |

### Learning Rate Schedule
- **Scheduler**: CosineAnnealingWarmRestarts
- **T_0**: 10 epochs (restart period)
- **T_mult**: 2 (period multiplier)
- **eta_min**: 1e-6 (minimum LR)

### Class Balancing
- **Method**: Weighted Cross-Entropy Loss
- **Weight Calculation**: Inverse class frequency
- **Implementation**: Automatic from training set distribution

### Cross-Validation
- **Strategy**: Stratified K-Fold
- **Folds**: 5
- **Random Seed**: 42 (for reproducibility)
- **Test Set**: 20% per fold (hold-out)

## Performance Optimization

### GPU Utilization
- **DataParallel**: Automatic multi-GPU distribution
- **Device IDs**: [0, 1] (customizable)
- **Pin Memory**: Enabled
- **Persistent Workers**: 16 processes
- **Prefetch Factor**: 4 batches

### Memory Management
- **Gradient Accumulation**: 1 step (adjustable)
- **Memory Cleanup Interval**: Every 25 batches
- **Automatic Mixed Precision**: Enabled
- **Dynamic Loss Scaling**: 2^16 initial scale

### CUDA Settings
```python
torch.backends.cudnn.benchmark = True       # Dynamic kernel selection
torch.backends.cudnn.deterministic = False  # Speed over reproducibility
```

## Evaluation Metrics

### Primary Metrics
1. **AUC-ROC**: Area under receiver operating characteristic curve
2. **95% CI**: Bootstrap confidence intervals (2000 samples)

### Secondary Metrics
1. **Youden Index Optimization**
   - Optimal cutoff via J-statistic
   - Sensitivity at optimal point
   - Specificity at optimal point

2. **F1 Score Optimization**
   - Optimal cutoff via F1 maximization
   - Precision-recall balance

### Statistical Analysis
- **Bootstrap Samples**: 2000
- **Confidence Level**: 95%
- **Random State**: 42 (reproducibility)

## Output Structure

```
output_dataparallel_optimized/
│
├── Model_1_BoundingBox/
│   ├── fold_1/
│   │   ├── metrics/
│   │   │   ├── epoch_metrics.json
│   │   │   ├── final_results.json
│   │   │   └── test_classification_report.json
│   │   ├── plots/
│   │   │   ├── learning_curves.png
│   │   │   ├── test_prediction_analysis.png
│   │   │   └── test_confusion_matrix.png
│   │   ├── predictions/
│   │   │   └── test_predictions.csv
│   │   ├── checkpoints/
│   │   │   └── epoch_{10,20,30}.pth
│   │   ├── final_model.pth
│   │   └── summary.txt
│   ├── fold_2-5/
│   │   └── [Same structure]
│   ├── model_summary.json
│   └── model_report.txt
│
├── Model_2_StandardVOI/
│   └── [Same structure]
│
├── Model_3_ExpandedVOI/
│   └── [Same structure]
│
└── model_comparison/
    └── model_comparison.csv
```

## Reproducibility Checklist

### Environment Setup
```bash
# 1. Create conda environment
conda create -n mri_classification python=3.9.16

# 2. Activate environment
conda activate mri_classification

# 3. Install PyTorch with CUDA
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 4. Install MONAI and dependencies
pip install monai==1.2.0
pip install -r requirements.txt
```

### Critical Parameters for Reproduction
✅ Random seed: 42 (all random operations)
✅ MONAI determinism: Enabled
✅ Target spacing: [1.0, 1.0, 2.0] mm
✅ Volume size: [128, 128, 80] voxels
✅ Batch size: 32
✅ Learning rate: 1e-4
✅ Training epochs: 30
✅ Cross-validation folds: 5
✅ Data augmentation probabilities: See table above

## Running the Code

### Basic Training
```python
from training_script import TrainingConfig, train_all_models_sequential

# Initialize configuration
config = TrainingConfig()

# Run training
results = train_all_models_sequential(config)

# Save comparison
save_final_comparison(results, config)
```

### Custom Configuration
```python
config = TrainingConfig(
    root_dir="/path/to/data",
    epochs=30,
    batch_size=32,
    learning_rate=1e-4,
    n_folds=5
)
```

## Acknowledgments

- MONAI Project for medical imaging deep learning framework
- PyTorch team for the deep learning platform
- NVIDIA for CUDA and mixed precision training support
