#!/usr/bin/env python3
"""
5-Fold Cross-Validation Training Script
- Bootstrap CI for AUC
- Training-derived Youden thresholds
- Complete separation: train threshold applied to test set
"""

import os
import sys
import gc
import warnings
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.functional import softmax

import matplotlib.pyplot as plt
import seaborn as sns

import monai
from monai.data import Dataset as MonaiDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Resized, ScaleIntensityd, MaskIntensityd,
    NormalizeIntensityd, RandFlipd, RandAffined,
    RandGaussianNoised, RandBiasFieldd, Rand3DElasticd,
    ToTensord
)
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism

warnings.filterwarnings("ignore")

# Set font to Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# CUDNN settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Training configuration"""
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
    weight_decay = 1e-5

    # Image settings
    target_spacing = (1.0, 1.0, 2.0)
    roi_size = (128, 128, 80)

    # Optimization
    use_mixed_precision = True
    gradient_clip_val = 1.0
    num_workers = 16

    # Statistics
    bootstrap_samples = 2000
    confidence_level = 0.95

    # Random seed
    seed = 42


# ============================================================================
# Utility Functions
# ============================================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     n_bootstrap: int = 2000, alpha: float = 0.05) -> Dict:
    """Calculate AUC with Bootstrap confidence interval"""
    np.random.seed(Config.seed)
    n_samples = len(y_true)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue

        try:
            auc = roc_auc_score(y_true[indices], y_prob[indices])
            bootstrap_aucs.append(auc)
        except:
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)
    ci_lower = np.percentile(bootstrap_aucs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha/2) * 100)

    return {
        'auc': roc_auc_score(y_true, y_prob),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_method': 'Bootstrap'
    }


def find_optimal_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Find optimal threshold using Youden's Index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)

    optimal_threshold = thresholds[optimal_idx]

    # Calculate metrics at optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'threshold': float(optimal_threshold),
        'youden_index': float(youden_index[optimal_idx]),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    }


def calculate_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                   threshold: float) -> Dict:
    """Calculate metrics at given threshold"""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'ppv': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


# ============================================================================
# Data Management
# ============================================================================
def collate_remove_none(batch):
    """Remove None samples from batch"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


class SafeDataset(MonaiDataset):
    """Dataset wrapper that handles transform errors"""
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"⚠️  Skipping sample {index}: {e}")
            return None


def get_transforms(is_training: bool = True) -> Compose:
    """Get data transforms"""
    keys = ["en_img", "t1_img", "t2_img", "en_lbl", "t1_lbl", "t2_lbl"]

    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        Spacingd(keys=keys, pixdim=Config.target_spacing,
                mode=["bilinear"]*3 + ["nearest"]*3),
        Resized(keys=keys, spatial_size=Config.roi_size,
               mode=["trilinear"]*3 + ["nearest"]*3),
        ScaleIntensityd(keys=["en_img", "t1_img", "t2_img"], minv=0.0, maxv=1.0),
        MaskIntensityd(keys=["en_img", "t1_img", "t2_img"], mask_key="en_lbl"),
        NormalizeIntensityd(keys=["en_img", "t1_img", "t2_img"], nonzero=True)
    ]

    if is_training:
        transforms += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandAffined(keys=keys, mode=["bilinear"]*3 + ["nearest"]*3,
                       prob=0.8, rotate_range=[0.1]*3,
                       translate_range=[10, 10, 5], scale_range=[0.1]*3),
            RandGaussianNoised(keys=["en_img", "t1_img", "t2_img"],
                             prob=0.3, mean=0.0, std=0.1),
            RandBiasFieldd(keys=["en_img", "t1_img", "t2_img"],
                          prob=0.2, degree=3, coeff_range=(0.0, 0.1)),
            Rand3DElasticd(keys=keys, prob=0.1, sigma_range=(5, 8),
                          magnitude_range=(100, 200),
                          spatial_size=Config.roi_size,
                          mode=["bilinear"]*3 + ["nearest"]*3)
        ]

    transforms.append(ToTensord(keys=["en_img", "t1_img", "t2_img"]))
    return Compose(transforms)


def load_labels() -> Dict[str, int]:
    """Load patient labels"""
    path = Path(Config.root_dir) / Config.label_file
    df = pd.read_csv(path)
    labels = df.set_index("patient_id")["label"].astype(int).to_dict()

    print(f"✅ Loaded {len(labels)} labels")
    unique, counts = np.unique(list(labels.values()), return_counts=True)
    print(f"   Class distribution: {dict(zip(unique, counts))}")

    return labels


def build_file_list(dataset_type: str, labels: Dict) -> List[Dict]:
    """Build file list for dataset"""
    base_dir = Path(Config.root_dir) / dataset_type

    # Find sequence directories
    seq_dirs = {}
    for d in base_dir.iterdir():
        if d.is_dir():
            name_lower = d.name.lower()
            if 'en' in name_lower:
                seq_dirs['en'] = d
            elif 't1' in name_lower:
                seq_dirs['t1'] = d
            elif 't2' in name_lower:
                seq_dirs['t2'] = d

    items = []
    for patient_dir in seq_dirs['en'].iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        if patient_id not in labels:
            continue

        # Find files
        def find_files(seq_dir):
            patient_path = seq_dir / patient_id
            if not patient_path.exists():
                return None, None

            files = list(patient_path.glob("*.nii.gz"))
            img, seg = None, None

            for f in files:
                if f.name.lower().startswith('seg'):
                    seg = str(f)
                else:
                    img = str(f)

            return img, seg

        en_img, en_seg = find_files(seq_dirs['en'])
        t1_img, _ = find_files(seq_dirs['t1'])
        t2_img, _ = find_files(seq_dirs['t2'])

        if all([en_img, t1_img, t2_img, en_seg]):
            items.append({
                "patient_id": patient_id,
                "en": en_img,
                "t1": t1_img,
                "t2": t2_img,
                "seg_en": en_seg
            })

    print(f"✅ Found {len(items)} complete datasets")
    return items


def prepare_data_dict(item: Dict, labels: Dict) -> Dict:
    """Prepare data dictionary for transform"""
    return {
        "en_img": item["en"],
        "t1_img": item["t1"],
        "t2_img": item["t2"],
        "en_lbl": item["seg_en"],
        "t1_lbl": item["seg_en"],
        "t2_lbl": item["seg_en"],
        "label": labels[item["patient_id"]],
        "patient_id": item["patient_id"]
    }


# ============================================================================
# Training Engine
# ============================================================================
class Trainer:
    """Training engine"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if Config.use_mixed_precision else None

    def create_model(self) -> nn.Module:
        """Create DenseNet121 model"""
        model = DenseNet121(spatial_dims=3, in_channels=3, out_channels=2)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"✅ Using DataParallel with {torch.cuda.device_count()} GPUs")

        model = model.to(self.device)
        return model

    def train_epoch(self, model: nn.Module, loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train one epoch"""
        model.train()
        total_loss = 0.0
        count = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            if batch is None:
                continue

            imgs = torch.cat([batch["en_img"], batch["t1_img"], batch["t2_img"]],
                           dim=1).to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            if Config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if Config.gradient_clip_val > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   Config.gradient_clip_val)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()

                if Config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   Config.gradient_clip_val)

                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

        return total_loss / count if count > 0 else 0.0

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Tuple[List, List]:
        """Evaluate model and return predictions"""
        model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                if batch is None:
                    continue

                imgs = torch.cat([batch["en_img"], batch["t1_img"], batch["t2_img"]],
                               dim=1).to(self.device)
                labels = batch["label"]

                if Config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(imgs)
                else:
                    outputs = model(imgs)

                probs = softmax(outputs, dim=1)[:, 1]

                y_true.extend(labels.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        return y_true, y_prob


# ============================================================================
# Training and Evaluation
# ============================================================================
def train_single_fold(model_name: str, fold: int, train_items: List,
                     test_items: List, labels: Dict, output_dir: Path) -> Dict:
    """Train and evaluate single fold"""
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{Config.n_folds}")
    print(f"{'='*60}")

    # Create output directory
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_transforms = get_transforms(is_training=True)
    test_transforms = get_transforms(is_training=False)

    train_data = [prepare_data_dict(item, labels) for item in train_items]
    test_data = [prepare_data_dict(item, labels) for item in test_items]

    train_ds = SafeDataset(data=train_data, transform=train_transforms)
    test_ds = SafeDataset(data=test_data, transform=test_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=True,
        collate_fn=collate_remove_none, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=8, pin_memory=True,
        collate_fn=collate_remove_none
    )

    # Create model and optimizer
    trainer = Trainer()
    model = trainer.create_model()
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=Config.learning_rate,
                                 weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Calculate class weights
    train_labels = [labels[item['patient_id']] for item in train_items]
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(trainer.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Train samples: {len(train_items)}, Test samples: {len(test_items)}")
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Training loop
    train_losses = []
    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")

        train_loss = trainer.train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f}, LR: {lr:.2e}")

    # Calculate optimal threshold from TRAINING data
    print("\n🎯 Calculating optimal threshold from training data...")
    train_y_true, train_y_prob = trainer.evaluate(model, train_loader)
    train_y_true = np.array(train_y_true)
    train_y_prob = np.array(train_y_prob)

    train_threshold_info = find_optimal_youden_threshold(train_y_true, train_y_prob)
    optimal_threshold = train_threshold_info['threshold']

    print(f"  Optimal Youden threshold: {optimal_threshold:.4f}")
    print(f"  Training sensitivity: {train_threshold_info['sensitivity']:.3f}")
    print(f"  Training specificity: {train_threshold_info['specificity']:.3f}")

    # Evaluate on TEST data using training threshold
    print("\n🔍 Evaluating on test data with training threshold...")
    test_y_true, test_y_prob = trainer.evaluate(model, test_loader)
    test_y_true = np.array(test_y_true)
    test_y_prob = np.array(test_y_prob)

    # Calculate AUC with Bootstrap CI
    auc_results = bootstrap_auc_ci(test_y_true, test_y_prob,
                                   n_bootstrap=Config.bootstrap_samples)

    # Calculate metrics at training threshold
    test_metrics = calculate_metrics_at_threshold(test_y_true, test_y_prob,
                                                  optimal_threshold)

    print(f"  Test AUC: {auc_results['auc']:.3f} "
          f"(95% CI: {auc_results['ci_lower']:.3f}-{auc_results['ci_upper']:.3f})")
    print(f"  Test sensitivity: {test_metrics['sensitivity']:.3f}")
    print(f"  Test specificity: {test_metrics['specificity']:.3f}")
    print(f"  Test accuracy: {test_metrics['accuracy']:.3f}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'y_true': test_y_true,
        'y_prob': test_y_prob,
        'y_pred': (test_y_prob >= optimal_threshold).astype(int),
        'threshold_used': optimal_threshold
    })
    predictions_df.to_csv(fold_dir / 'test_predictions.csv', index=False)

    # Save results
    results = {
        'fold': fold,
        'train_threshold': optimal_threshold,
        'train_threshold_info': train_threshold_info,
        'test_auc': auc_results['auc'],
        'test_ci_lower': auc_results['ci_lower'],
        'test_ci_upper': auc_results['ci_upper'],
        'test_metrics': test_metrics,
        'n_train': len(train_items),
        'n_test': len(test_items)
    }

    with open(fold_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot ROC curve
    plot_roc_curve(test_y_true, test_y_prob, auc_results,
                  fold_dir / 'roc_curve.png', fold)

    # Cleanup
    del model, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return results


def plot_roc_curve(y_true, y_prob, auc_results, save_path, fold):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5,
            label=f'ROC Curve (AUC = {auc_results["auc"]:.3f}, '
                  f'95% CI: {auc_results["ci_lower"]:.3f}-{auc_results["ci_upper"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Chance')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title(f'ROC Curve - Fold {fold}', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    plt.gca().tick_params(width=1.5, labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """Aggregate results across folds"""
    aucs = [r['test_auc'] for r in fold_results]
    thresholds = [r['train_threshold'] for r in fold_results]
    sensitivities = [r['test_metrics']['sensitivity'] for r in fold_results]
    specificities = [r['test_metrics']['specificity'] for r in fold_results]
    accuracies = [r['test_metrics']['accuracy'] for r in fold_results]

    aggregated = {
        'mean_test_auc': np.mean(aucs),
        'std_test_auc': np.std(aucs),
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'mean_sensitivity': np.mean(sensitivities),
        'std_sensitivity': np.std(sensitivities),
        'mean_specificity': np.mean(specificities),
        'std_specificity': np.std(specificities),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'fold_results': fold_results
    }

    return aggregated


def train_model(model_name: str, dataset_type: str, labels: Dict, output_dir: Path):
    """Train model with 5-fold cross-validation"""
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")

    model_dir = output_dir / model_name.replace(' ', '_')
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    items = build_file_list(dataset_type, labels)

    # Prepare stratified k-fold
    patient_ids = [item['patient_id'] for item in items]
    y = np.array([labels[pid] for pid in patient_ids])

    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True,
                         random_state=Config.seed)

    # Train each fold
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, y), 1):
        train_items = [items[i] for i in train_idx]
        test_items = [items[i] for i in test_idx]

        fold_result = train_single_fold(model_name, fold, train_items,
                                       test_items, labels, model_dir)
        fold_results.append(fold_result)

    # Aggregate results
    aggregated = aggregate_fold_results(fold_results)

    # Save aggregated results
    with open(model_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Test AUC: {aggregated['mean_test_auc']:.3f} ± {aggregated['std_test_auc']:.3f}")
    print(f"Optimal Threshold: {aggregated['mean_threshold']:.3f} ± {aggregated['std_threshold']:.3f}")
    print(f"Test Sensitivity: {aggregated['mean_sensitivity']:.3f} ± {aggregated['std_sensitivity']:.3f}")
    print(f"Test Specificity: {aggregated['mean_specificity']:.3f} ± {aggregated['std_specificity']:.3f}")
    print(f"Test Accuracy: {aggregated['mean_accuracy']:.3f} ± {aggregated['std_accuracy']:.3f}")

    return aggregated


def save_optimal_thresholds(all_results: Dict, output_dir: Path):
    """Save optimal thresholds for external validation"""
    thresholds = {}

    for model_name, results in all_results.items():
        thresholds[model_name] = results['mean_threshold']

    threshold_file = output_dir / 'optimal_thresholds.json'
    with open(threshold_file, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✅ Optimal thresholds saved to: {threshold_file}")
    print("\nOptimal Thresholds (mean across 5 folds):")
    for model_name, threshold in thresholds.items():
        print(f"  {model_name}: {threshold:.4f}")


def create_comparison_table(all_results: Dict, output_dir: Path):
    """Create comparison table for all models"""
    data = []

    for model_name, results in all_results.items():
        data.append({
            'Model': model_name,
            'Test AUC': f"{results['mean_test_auc']:.3f} ± {results['std_test_auc']:.3f}",
            'Threshold': f"{results['mean_threshold']:.3f} ± {results['std_threshold']:.3f}",
            'Sensitivity': f"{results['mean_sensitivity']:.3f} ± {results['std_sensitivity']:.3f}",
            'Specificity': f"{results['mean_specificity']:.3f} ± {results['std_specificity']:.3f}",
            'Accuracy': f"{results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}"
        })

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)

    # Save markdown
    md_path = output_dir / 'model_comparison.md'
    with open(md_path, 'w') as f:
        f.write("# 5-Fold Cross-Validation Results\n\n")
        f.write("**AUC confidence intervals calculated using Bootstrap method (2000 samples)**\n\n")
        f.write("**Thresholds optimized on training data using Youden's Index**\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n---\n")
        f.write("Results represent mean ± standard deviation across 5 folds\n")

    print(f"\n✅ Comparison table saved to: {csv_path}")
    print(f"✅ Markdown report saved to: {md_path}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution"""
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION TRAINING")
    print("="*80)
    print(f"Output directory: {Config.output_dir}")
    print(f"Models: {list(Config.model_configs.keys())}")
    print(f"Folds: {Config.n_folds}")
    print(f"Epochs: {Config.epochs}")
    print(f"Bootstrap samples: {Config.bootstrap_samples}")
    print("="*80 + "\n")

    # Set seed
    set_seed(Config.seed)

    # Create output directory
    output_dir = Path(Config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels = load_labels()

    # Train all models
    all_results = {}
    for model_name, dataset_type in Config.model_configs.items():
        results = train_model(model_name, dataset_type, labels, output_dir)
        all_results[model_name] = results

    # Save optimal thresholds for external validation
    save_optimal_thresholds(all_results, output_dir)

    # Create comparison table
    create_comparison_table(all_results, output_dir)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext step: Use optimal_thresholds.json for external validation")


if __name__ == "__main__":
    main()
