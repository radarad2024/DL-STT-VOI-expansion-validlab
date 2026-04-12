# Reproduction Parameters Documentation
## Complete Parameter Specification for Scientific Reproducibility

> **Influence of Voxel-of-Interest Expansion on Deep Learning-based Classification of Soft Tissue Tumors on MRI**

*This document provides complete parameter specification for reproducing the experimental results reported in the paper.*

*Last updated: April 12, 2026*

---

### Table of Contents
1. [Study Design](#1-study-design)
2. [Data Preprocessing Parameters](#2-data-preprocessing-parameters)
3. [Model Architecture Parameters](#3-model-architecture-parameters)
4. [Training Hyperparameters](#4-training-hyperparameters)
5. [Augmentation Parameters](#5-augmentation-parameters)
6. [Evaluation Parameters](#6-evaluation-parameters)
7. [System Configuration](#7-system-configuration)
8. [Framework Versions](#8-framework-versions)
9. [Computational Resources](#9-computational-resources)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. Study Design

### 1.1 Dataset Summary
| | Training Set | External Test Set |
|:---|:---:|:---:|
| Source | Institution 1 | Institution 2 |
| Period | Jan 2009 – Aug 2019 | Mar 2016 – Jun 2025 |
| Total patients | 125 | 58 |
| Benign | 71 (56.8%) | 38 (65.5%) |
| Malignant | 54 (43.2%) | 20 (34.5%) |
| Scanner | 3.0 T (Verio; Siemens) | 3.0 T (Verio; Siemens) |

### 1.2 VOI Definitions
| VOI Type | Abbreviation | Model | Description |
|:---|:---:|:---:|:---|
| Rectangular | R-VOI | Model 1 | 3D bounding box tightly enclosing the tumor, axis-aligned on axial images |
| Standard | S-VOI | Model 2 | Manual segmentation of the tumor boundary (ITK-SNAP v3.8.0) |
| Expanded | E-VOI | Model 3 | S-VOI dilated by 1 cm isotropically to include peritumoral tissue |

### 1.3 MRI Sequences
| Channel | Sequence | Abbreviation | Role |
|:---:|:---|:---:|:---|
| 1 | Contrast-enhanced fat-suppressed T1WI | CE-FS-T1WI | Vascularized components, necrosis |
| 2 | T1-weighted imaging | T1WI | Anatomic detail, fat planes |
| 3 | Fat-suppressed T2-weighted imaging | T2WI | Water content, myxoid stroma |

### 1.4 Multi-Sequence Fusion
| Parameter | Value | Details |
|:---|:---:|:---|
| Sequence order | [EN, T1, T2] | Channel concatenation |
| Channel dimension | 1 | After batch dimension |
| Fusion method | Early fusion | Pre-network concatenation |
| Alignment | Co-registered | Assumed pre-aligned |

---

## 2. Data Preprocessing Parameters

### 2.1 Image Loading and Formatting
| Parameter | Value | Framework | Function | Rationale |
|:---|:---:|:---:|:---|:---|
| Image format | NIfTI (.nii.gz) | MONAI | `LoadImaged` | Medical imaging standard |
| Loader backend | ITK | MONAI | — | Robust orientation handling |
| Image only | True | MONAI | `LoadImaged` | Exclude metadata |
| Channel position | First | MONAI | `EnsureChannelFirstd` | PyTorch convention |
| Data type | float32 | NumPy/PyTorch | — | Precision vs memory balance |

### 2.2 Spatial Resampling
| Parameter | Value | Method | Implementation |
|:---|:---:|:---|:---|
| Target spacing (x, y, z) | [1.0, 1.0, 2.0] mm | Trilinear / Nearest | `MONAI Spacingd` |
| Image interpolation | Bilinear | B-spline order 1 | `mode="bilinear"` |
| Mask interpolation | Nearest neighbor | Order 0 | `mode="nearest"` |
| Padding mode | Border | Constant extension | Default MONAI |
| Anti-aliasing | Enabled | Gaussian smoothing | Built-in MONAI |

### 2.3 Volume Standardization
| Parameter | Value | Justification |
|:---|:---:|:---|
| Target size (x, y, z) | [128, 128, 80] voxels | GPU memory optimization |
| Resize mode (images) | Trilinear | Smooth interpolation |
| Resize mode (masks) | Nearest | Preserve binary labels |
| Align corners | False | PyTorch default |
| Preserve range | True | Maintain intensity values |

### 2.4 Intensity Normalization Pipeline

Intensities were normalized using a custom Z-score method applied to the non-zero (masked) region:

```
f(x) = (x − μ_nonzero) / σ_nonzero × 1000
```

| Step | Method | Parameters | Formula |
|:---:|:---|:---|:---|
| 1 | Mask application | `MaskIntensityd` | x × mask (zero outside VOI) |
| 2 | Compute statistics | Non-zero mean/std | μ, σ over voxels > 0 |
| 3 | Z-score & scale | Custom `PaperNormalized` | (x − μ) / σ × 1000 |

This is implemented as `PaperNormalized`, a custom MONAI `MapTransform` applied to all three image channels after masking.

---

## 3. Model Architecture Parameters

### 3.1 DenseNet-121 Configuration
| Component | Value | Details |
|:---|:---:|:---|
| Base architecture | DenseNet-121 | 3D variant |
| Framework | MONAI | `monai.networks.nets.DenseNet121` |
| Spatial dimensions | 3 | Volumetric convolutions |
| Input channels | 3 | Concatenated sequences |
| Output classes | 2 | Binary classification |
| Growth rate | 32 | Feature maps per layer |
| Block configuration | [6, 12, 24, 16] | Layers per dense block |
| Compression factor | 0.5 | Transition layer reduction |
| Initial features | 64 | First convolution output |
| Dropout rate | 0.0 | No dropout (small dataset) |

### 3.2 Network Layer Details
| Layer type | Kernel size | Stride | Padding | Activation |
|:---|:---:|:---:|:---:|:---:|
| Initial Conv3D | 7×7×7 | 2×2×2 | 3×3×3 | ReLU |
| Initial MaxPool3D | 3×3×3 | 2×2×2 | 1×1×1 | — |
| Dense block Conv3D | 3×3×3 | 1×1×1 | 1×1×1 | ReLU |
| Transition Conv3D | 1×1×1 | 1×1×1 | 0×0×0 | ReLU |
| Transition AvgPool3D | 2×2×2 | 2×2×2 | 0×0×0 | — |
| Global AdaptiveAvgPool3D | Adaptive | — | — | — |
| Classifier (Linear) | — | — | — | Softmax |

### 3.3 Batch Normalization
| Parameter | Value |
|:---|:---:|
| Epsilon | 1×10⁻⁵ |
| Momentum | 0.1 |
| Affine | True |
| Track running stats | True |

---

## 4. Training Hyperparameters

### 4.1 Optimization Configuration
| Parameter | Value | Justification |
|:---|:---:|:---|
| Optimizer | AdamW | Weight decay decoupling |
| Base learning rate | 5×10⁻⁵ | Stable convergence for small dataset |
| β₁ | 0.9 | Default momentum |
| β₂ | 0.999 | Default RMSprop |
| ε | 1×10⁻⁸ | Numerical stability |
| Weight decay | 1×10⁻⁵ | L2 regularization |
| AMSGrad | False | Standard Adam |

### 4.2 Learning Rate Schedule
| Parameter | Value | Details |
|:---|:---:|:---|
| Scheduler | CosineAnnealingWarmRestarts | Cyclical learning |
| T₀ | 10 epochs | Initial period |
| T_mult | 2 | Period doubling |
| η_min | 0 | Default minimum LR |
| Last epoch | −1 | Start from beginning |

The learning rate follows a cosine annealing curve that restarts at epochs 10 and 30 (T₀=10, T_mult=2).

### 4.3 Training Configuration
| Parameter | Value | Details |
|:---|:---:|:---|
| Total epochs | 30 | Fixed training |
| Batch size | 32 | Total across GPUs |
| Effective batch / GPU | 16 | With 2 GPUs |
| Gradient accumulation | 1 | No accumulation |
| Early stopping | Disabled | Fixed 30 epochs |
| Checkpoint epochs | 10, 20, 30 | Saved for analysis |

### 4.4 Loss Function
| Parameter | Value | Implementation |
|:---|:---:|:---|
| Loss type | Cross-entropy | `nn.CrossEntropyLoss` |
| Class weights | Inverse frequency | `n_total / (n_classes × n_class)` |
| Label smoothing | 0.0 | No smoothing |
| Reduction | Mean | Average over batch |
| Ignore index | −100 | Default PyTorch |

---

## 5. Augmentation Parameters

Augmentation was applied **only during training**. Validation and external test sets used no augmentation.

### 5.1 Spatial Augmentations
| Transform | Probability | Parameters | Range / Values |
|:---|:---:|:---|:---|
| Random flip X | 0.5 | `spatial_axis=0` | Binary flip |
| Random flip Y | 0.5 | `spatial_axis=1` | Binary flip |
| Random flip Z | 0.0 | Disabled | Anatomical constraint |

### 5.2 Affine Transformations
| Component | Probability | Range | Units |
|:---|:---:|:---:|:---:|
| Overall probability | 0.8 | — | — |
| Rotation (x, y, z) | — | [−0.1, 0.1] | Radians |
| Translation X | — | [−10, 10] | mm |
| Translation Y | — | [−10, 10] | mm |
| Translation Z | — | [−5, 5] | mm |
| Scale (x, y, z) | — | [−0.1, 0.1] | Fraction |
| Shear | — | 0.0 | Disabled |
| Image interpolation | — | Bilinear | `mode="bilinear"` |
| Mask interpolation | — | Nearest | `mode="nearest"` |

### 5.3 Intensity Augmentations
| Transform | Probability | Parameters | Distribution |
|:---|:---:|:---|:---|
| Gaussian noise | 0.3 | mean=0, std=0.1 | N(0, 0.01) |
| Bias field | 0.2 | degree=3, coeff=[0.0, 0.1] | Polynomial |
| Intensity shift | 0.0 | Disabled | — |
| Intensity scale | 0.0 | Disabled | — |

### 5.4 Elastic Deformation
| Parameter | Value | Range |
|:---|:---:|:---|
| Probability | 0.1 | — |
| Sigma range | [5, 8] | Gaussian kernel |
| Magnitude range | [100, 200] | Displacement |
| Grid size | [128, 128, 80] | Same as input |
| Image interpolation | Bilinear | `mode="bilinear"` |
| Mask interpolation | Nearest | `mode="nearest"` |
| Padding mode | Border | Constant extension |

---

## 6. Evaluation Parameters

### 6.1 Cross-Validation
| Parameter | Value | Implementation |
|:---|:---:|:---|
| Strategy | Stratified K-fold | `sklearn.model_selection.StratifiedKFold` |
| Number of folds | 5 | 80/20 split per fold |
| Shuffle | True | Random permutation |
| Random state | 42 | Reproducibility |

### 6.2 Performance Metrics
| Metric | Method | Parameters |
|:---|:---|:---|
| AUC-ROC | Trapezoidal rule | `sklearn.metrics.roc_auc_score` |
| Youden index | max(TPR − FPR) | All thresholds |
| Sensitivity | TP / (TP + FN) | At optimal cutoff |
| Specificity | TN / (TN + FP) | At optimal cutoff |
| F1 score | 2TP / (2TP + FP + FN) | At optimal cutoff |

### 6.3 Confidence Intervals
| Context | Method | Parameters |
|:---|:---|:---|
| Internal AUC | Bootstrap percentile | n=2,000 resamples, α=0.05 |
| External AUC | Bootstrap percentile | n=2,000 resamples, α=0.05 |
| External sens/spec | Exact binomial (Clopper–Pearson) | α=0.05 |
| AUC comparison | DeLong's test | Paired comparison |

### 6.4 Statistical Analysis
| Parameter | Value | Method |
|:---|:---:|:---|
| Bootstrap samples | 2,000 | With replacement |
| Confidence level | 95% | Two-tailed |
| Random seed | 42 | NumPy random |
| Stratification | Maintained | In bootstrap |
| CI method | Percentile | [2.5%, 97.5%] |
| Statistical software | R 4.0.0 | DeLong test |

### 6.5 External Validation Threshold
The sensitivity/specificity cutoff for external validation was **predetermined from the training set** by selecting the threshold that achieved a sensitivity of at least 80% with maximum specificity.

---

## 7. System Configuration

### 7.1 Hardware Specifications
| Component | Specification | Configuration |
|:---|:---|:---|
| GPU model | NVIDIA RTX 4080 | 2× units |
| GPU memory | 16 GB GDDR6X | Per GPU |
| GPU driver | 535.154.05 | Linux |
| CPU model | Intel Core i9-14900K | 24 cores |
| System RAM | 64 GB DDR5-5600 | Dual channel |
| Storage | NVMe SSD | 2 TB capacity |

### 7.2 CUDA Configuration
| Parameter | Value | Purpose |
|:---|:---:|:---|
| CUDA version | 11.8 | Compatibility |
| cuDNN version | 8.7.0 | Deep learning |
| `cudnn.benchmark` | True | Dynamic kernels |
| CUDA deterministic | False | Speed priority |
| CUDA backend | cuBLAS | Matrix operations |

### 7.3 DataLoader Configuration
| Parameter | Training | Validation | Justification |
|:---|:---:|:---:|:---|
| Number of workers | 8 | 8 | CPU parallelism |
| Pin memory | True | True | GPU transfer |
| Drop last | True | False | Batch consistency |
| Shuffle | True | False | Random / reproducible |
| Collate function | Custom | Custom | Skip None (corrupted samples) |

### 7.4 Mixed Precision Training
| Parameter | Value | Details |
|:---|:---:|:---|
| Enabled | True | Memory efficiency |
| Initial scale | 2¹⁶ | 65,536 |
| Growth factor | 2.0 | Scale increase |
| Backoff factor | 0.5 | Scale decrease |
| Growth interval | 2,000 | Steps between increases |
| AMP backend | Native PyTorch | `torch.cuda.amp` |

---

## 8. Framework Versions

### 8.1 Core Dependencies
| Package | Version | Source |
|:---|:---:|:---|
| Python | 3.9.16 | Official |
| PyTorch | 2.0.1+cu118 | PyTorch.org |
| TorchVision | 0.15.2+cu118 | PyTorch.org |
| MONAI | 1.2.0 | PyPI |
| NumPy | 1.24.3 | PyPI |
| Pandas | 2.0.2 | PyPI |

### 8.2 Scientific Computing
| Package | Version | Usage |
|:---|:---:|:---|
| scikit-learn | 1.3.0 | Cross-validation, metrics |
| SciPy | 1.10.1 | Statistical analysis |
| Matplotlib | 3.7.1 | Visualization |
| Seaborn | 0.12.2 | Statistical plots |

### 8.3 Medical Imaging & Utilities
| Package | Version | Purpose |
|:---|:---:|:---|
| SimpleITK | 2.2.1 | Medical imaging I/O |
| nibabel | 5.1.0 | NIfTI support |
| ITK-SNAP | 3.8.0 | Segmentation (external tool) |
| tqdm | 4.65.0 | Progress bars |
| Pillow | 9.5.0 | Image I/O |

---

## 9. Computational Resources

### 9.1 Training Time Estimates (per model)
| Component | Time | Details |
|:---|:---:|:---|
| Single epoch | ~3.5 min | Full dataset |
| Single fold (30 epochs) | ~105 min | Including evaluation |
| 5-fold CV | ~8.75 hours | Complete validation |
| Final model (30 epochs) | ~105 min | All data |
| **Total per VOI type** | **~10.5 hours** | CV + final model |
| **Three models total** | **~31.5 hours** | Full experiment |

### 9.2 Memory Requirements
| Stage | GPU Memory | System RAM |
|:---|:---:|:---:|
| Training | ~14 GB per GPU | ~32 GB |
| Inference | ~6 GB | ~16 GB |
| Peak usage | ~15 GB per GPU | ~48 GB |

---

## 10. Verification Checklist

### Random Seed Configuration

All random seeds are set to **42** for full reproducibility:

```python
SEED = 42

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from monai.utils import set_determinism
set_determinism(seed=SEED)

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### Pre-Training Verification
- [ ] Dataset structure matches specification (EN/T1/T2 subdirectories)
- [ ] All patient IDs have complete sequences and segmentation masks
- [ ] Labels CSV properly formatted (`patient_id,label`)
- [ ] GPU drivers and CUDA 11.8 installed
- [ ] Python environment activated with correct package versions
- [ ] All dependencies installed (`monai==1.2.0`, `torch==2.0.1+cu118`)
- [ ] Sufficient disk space for checkpoints (~2 GB per model)

### During Training Verification
- [ ] Loss is decreasing (non-monotonic is expected with warm restarts)
- [ ] No NaN or inf values in loss
- [ ] GPU utilization > 90%
- [ ] Memory usage stable (~14 GB per GPU)
- [ ] Checkpoint files being saved at epochs 10, 20, 30
- [ ] No data loading bottlenecks (workers ≥ 8)

### Post-Training Verification
- [ ] All 5 folds completed successfully
- [ ] `training_results.json` saved in output directory
- [ ] Final model checkpoints exist for epochs 10, 20, 30
- [ ] AUC values within expected range (see below)
- [ ] Model files loadable with `torch.load()`
- [ ] Results reproducible with same seed (±0.02 AUC)

### Expected Results (Reference)

| | R-VOI | S-VOI | E-VOI |
|:---|:---:|:---:|:---:|
| **Internal AUC** | 0.775 | 0.769 | 0.789 |
| **External AUC** | 0.834 | 0.816 | 0.849 |

Minor variations (±0.02 AUC) are expected due to hardware-level non-determinism in CUDA operations.

---

*This document provides complete parameter specification for reproducing the experimental results reported in the paper "Influence of Voxel-of-Interest Expansion on Deep Learning-based Classification of Soft Tissue Tumors on MRI"*
