# DL-STT-VOI-Extension

Code for **"Influence of Volume-of-Interest Extension on Deep Learning-based Classification of Soft Tissue Tumors on MRI."**

DenseNet-121 models classify soft tissue tumors as non-malignant or malignant on multi-sequence MRI (T1WI, T2WI, and contrast-enhanced T1WI). Three volume-of-interest (VOI) strategies are compared to evaluate the effect of including the peritumoral region.

| VOI | Description |
|-----|-------------|
| **R-VOI** | Rectangular VOI: the smallest rectangular box enclosing the tumor |
| **S-VOI** | Standard VOI: the tumor segmentation without any margin |
| **E-VOI** | Extended VOI: the tumor segmentation extended by 1 cm into the peritumoral tissue |

## Repository structure

```
code/
├── make_voi_masks.py        # R-VOI and E-VOI masks from the tumor segmentation
├── preprocessing.py         # preprocessing transforms and data loading (shared)
├── internal_validation.py   # 5-fold cross-validation, final models, cutoffs, ROC curves
└── external_validation.py   # inference on the external test set, statistics, ROC curves
```

## VOI masks

`code/make_voi_masks.py` generates the R-VOI and E-VOI masks from the tumor segmentation (S-VOI).

- **R-VOI**: the smallest rectangular box enclosing the tumor segmentation, aligned with the image axes.
- **E-VOI**: the tumor segmentation extended by 1 cm. The extension is performed within each axial slice, so the E-VOI covers the same slices as the segmentation.

## Preprocessing

The same steps are applied to all three VOI models (`code/preprocessing.py`).

1. Load the three sequences and the VOI mask.
2. Resample the whole image volume to 1 × 1 × 2 mm.
3. Resize the whole image volume to 128 × 128 × 80 voxels.
4. Apply the VOI mask (voxels outside the VOI are set to 0). The images are not cropped to the VOI.
5. Normalize intensities within the VOI: f(x) = (x − μ) / σ × 1000.
6. Training only: augmentation applied on the fly each time a case is loaded (random flipping, affine transformation, Gaussian noise, bias field, and elastic deformation). No augmentation is used for the validation folds or the external test set.

## Data

Patient data are not included in this repository. The S-VOI directory (images and tumor segmentation) is prepared first; the R-VOI and E-VOI directories are created from it with `make_voi_masks.py`.

```
data/
├── train/                         # training set
│   ├── labels.csv                 # patient_id, label
│   ├── R-VOI/
│   │   ├── EN/<patient_id>/       # contrast-enhanced T1WI: image.nii.gz, seg_label.nii.gz (VOI mask)
│   │   ├── T1/<patient_id>/       # T1WI: image.nii.gz
│   │   └── T2/<patient_id>/       # T2WI: image.nii.gz
│   ├── S-VOI/                     # same layout, S-VOI mask
│   └── E-VOI/                     # same layout, E-VOI mask
└── external/                      # external test set
    ├── labels.csv
    ├── readers.csv                # patient_id, Reader1, Reader2 (5-point score, 0–4)
    ├── R-VOI/ ...
    ├── S-VOI/ ...
    └── E-VOI/ ...
```

- `labels.csv`: `label` is 0 for non-malignant (benign or intermediate) and 1 for malignant tumors.
- The VOI mask file name starts with `seg` and is placed in the `EN` folder of each patient.

## Usage

```bash
pip install torch monai nibabel scikit-learn scipy pandas matplotlib

# 1. generate the R-VOI and E-VOI masks from the tumor segmentation (S-VOI)
python code/make_voi_masks.py --src ./data/train/S-VOI --dst_root ./data/train
python code/make_voi_masks.py --src ./data/external/S-VOI --dst_root ./data/external

# 2. (optional) check the preprocessing of one VOI directory
python code/preprocessing.py --data_dir ./data/train/E-VOI --labels ./data/train/labels.csv

# 3. internal validation: 5-fold cross-validation, final models, and cutoffs
python code/internal_validation.py --data_root ./data/train \
    --labels ./data/train/labels.csv --output_dir ./outputs

# 4. external validation with the final models and the predetermined cutoffs
python code/external_validation.py --data_root ./data/external \
    --labels ./data/external/labels.csv --readers ./data/external/readers.csv \
    --model_dir ./outputs --output_dir ./outputs/external
```

Run the scripts from the repository root, or from `code/` with the paths adjusted.

## Model and training

| Parameter | Value |
|-----------|-------|
| Architecture | DenseNet-121, 3D (MONAI) |
| Input | 3 channels (contrast-enhanced T1WI, T1WI, T2WI), 128 × 128 × 80 voxels |
| Voxel spacing | 1 × 1 × 2 mm |
| Optimizer | AdamW (weight decay 1e-5) |
| Learning rate | 5 × 10⁻⁵ |
| Scheduler | Cosine annealing with warm restarts (T₀ = 10, T_mult = 2) |
| Batch size | 32 |
| Epochs | 30 |
| Loss | Cross-entropy with class weights inversely proportional to class frequency |
| Mixed precision | Enabled |
| Random seed | 42 |

## Internal validation (`internal_validation.py`)

- Stratified 5-fold cross-validation at the patient level.
- AUC of each fold and mean AUC; 95% CI by bootstrapping (2,000 resamples).
- Sensitivity and specificity at the Youden-optimal cutoff of each validation fold.
- Paired t-tests of the fold AUCs between models.
- Final models are trained on the entire training set.
- One cutoff per model is determined from the pooled out-of-fold predictions of the training set (sensitivity ≥ 80% with maximum specificity) and saved to `thresholds.json`.

Outputs: `cv_results.json`, `<VOI>/fold*.pth`, `<VOI>/final.pth`, `thresholds.json`, `internal_performance.csv`, `internal_auc_comparison.csv`, `Fig3a_Internal_ROC.png/.tiff`.

Cutoffs used in the study: R-VOI 0.3452, S-VOI 0.3749, E-VOI 0.3771.

## External validation (`external_validation.py`)

- The final models are applied to the external test set with the same preprocessing as in training (no augmentation).
- The cutoffs predetermined from the training set are applied without modification; a score of ≥ 3 is used for the radiologists.
- AUC with DeLong 95% CI; sensitivity and specificity with exact binomial (Clopper–Pearson) 95% CI.
- Pairwise DeLong tests (models vs. models, models vs. readers, reader vs. reader).

Outputs: `external_predictions.csv`, `external_performance.csv`, `external_delong_tests.csv`, `Fig3b_External_ROC.png/.tiff`.

## Requirements

```
torch >= 1.12
monai >= 1.0
nibabel
scikit-learn
scipy
pandas
numpy
matplotlib
```

## License

This project is for academic research purposes.
