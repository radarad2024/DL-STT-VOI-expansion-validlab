"""Internal validation: stratified 5-fold cross-validation on the training set.

DenseNet-121 for soft tissue tumor classification on multi-sequence MRI with
three VOI strategies: R-VOI (rectangular), S-VOI (standard segmentation), and
E-VOI (1-cm peritumoral extension).

Steps
  1. 5-fold cross-validation at the patient level for each VOI model.
  2. Final models trained on the entire training set (used for external validation).
  3. Cutoff per model from the pooled out-of-fold predictions
     (sensitivity >= 80% with maximum specificity), saved for external validation.
  4. ROC curves and summary of internal performance.

Usage
    python internal_validation.py --data_root ./data/train \
        --labels ./data/train/labels.csv --output_dir ./outputs
"""
import os
import gc
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import (
    VOI_NAMES, get_transforms, load_labels, build_file_list, prepare_dict,
    SafeDataset, collate_fn,
)

# ── Hyperparameters ──
SEED       = 42
N_FOLDS    = 5
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 5e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Train & evaluate ──
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, count = 0.0, 0
    for batch in loader:
        if batch is None: continue
        imgs = torch.cat([batch["en_img"], batch["t1_img"], batch["t2_img"]], dim=1).to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total_loss / count if count > 0 else 0


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        if batch is None: continue
        imgs = torch.cat([batch["en_img"], batch["t1_img"], batch["t2_img"]], dim=1).to(device)
        probs = softmax(model(imgs), dim=1)[:, 1].cpu().tolist()
        y_true.extend(batch["label"].tolist())
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_prob)


def build_model():
    model = DenseNet121(spatial_dims=3, in_channels=3, out_channels=2).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def build_training(model, y):
    """Loss with inverse class-frequency weights, optimizer, scheduler, and AMP scaler."""
    counts = np.bincount(y)
    weights = torch.FloatTensor(len(y) / (len(counts) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, scheduler, scaler


def save_state(model, path):
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, path)


# ── 1. 5-fold cross-validation ──
def run_cross_validation(data_dirs, labels, output_dir, num_workers):
    all_cv_results = {}
    for voi_name, data_dir in data_dirs.items():
        print(f"\n{'='*60}\n{voi_name}\n{'='*60}")
        items = build_file_list(data_dir, labels)
        pids = [it["patient_id"] for it in items]
        y = [labels[pid] for pid in pids]

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(pids, y), 1):
            print(f"  Fold {fold}/{N_FOLDS}")
            train_items = [items[i] for i in train_idx]
            val_items = [items[i] for i in val_idx]

            train_ds = SafeDataset([prepare_dict(it, labels) for it in train_items], get_transforms(True))
            val_ds = SafeDataset([prepare_dict(it, labels) for it in val_items], get_transforms(False))
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                                      pin_memory=True, collate_fn=collate_fn, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                                    pin_memory=True, collate_fn=collate_fn)

            model = build_model()
            criterion, optimizer, scheduler, scaler = build_training(
                model, [labels[it["patient_id"]] for it in train_items])

            for epoch in range(1, EPOCHS + 1):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
                scheduler.step()

            y_true, y_prob = evaluate(model, val_loader)
            auc = roc_auc_score(y_true, y_prob)
            print(f"    AUC: {auc:.4f}")
            fold_results.append({"fold": fold, "y_true": y_true.tolist(), "y_prob": y_prob.tolist(), "auc": auc})

            os.makedirs(f"{output_dir}/{voi_name}", exist_ok=True)
            save_state(model, f"{output_dir}/{voi_name}/fold{fold}.pth")

            del model, train_loader, val_loader
            gc.collect(); torch.cuda.empty_cache()

        all_cv_results[voi_name] = fold_results
        print(f"  Mean AUC: {np.mean([f['auc'] for f in fold_results]):.3f}")
    return all_cv_results


# ── 2. Final models trained on the entire training set ──
def train_final_models(data_dirs, labels, output_dir, num_workers):
    for voi_name, data_dir in data_dirs.items():
        print(f"\nFinal model: {voi_name}")
        items = build_file_list(data_dir, labels)
        y = [labels[it["patient_id"]] for it in items]

        ds = SafeDataset([prepare_dict(it, labels) for it in items], get_transforms(True))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn, drop_last=True)

        model = build_model()
        criterion, optimizer, scheduler, scaler = build_training(model, y)

        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, loader, optimizer, criterion, scaler)
            scheduler.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: loss={loss:.4f}")

        os.makedirs(f"{output_dir}/{voi_name}", exist_ok=True)
        save_state(model, f"{output_dir}/{voi_name}/final.pth")

        del model, loader
        gc.collect(); torch.cuda.empty_cache()


# ── 3. Cutoff for external validation ──
def pooled_predictions(folds):
    y_true = np.concatenate([np.array(f["y_true"]) for f in folds])
    y_prob = np.concatenate([np.array(f["y_prob"]) for f in folds])
    return y_true, y_prob


def determine_thresholds(all_cv_results):
    """One cutoff per model from the pooled out-of-fold predictions:
    sensitivity >= 80% with maximum specificity."""
    thresholds = {}
    for voi_name, folds in all_cv_results.items():
        all_yt, all_yp = pooled_predictions(folds)
        fpr, tpr, thr = roc_curve(all_yt, all_yp)
        candidates = [(t, tpr[i], 1-fpr[i]) for i, t in enumerate(thr) if tpr[i] >= 0.80]
        best = max(candidates, key=lambda x: x[2])
        thresholds[voi_name] = float(best[0])
        print(f"  {voi_name}: cutoff={best[0]:.6f} "
              f"(pooled out-of-fold sensitivity={best[1]:.3f}, specificity={best[2]:.3f})")
    return thresholds


# ── 4. ROC curves and summary ──
def plot_internal_roc(all_cv_results, output_dir):
    """Mean ROC curve of the five folds (+/- 1 SD) and bootstrap 95% CI of the AUC."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 11})
    colors = {"R-VOI": "#E53935", "S-VOI": "#43A047", "E-VOI": "#1E88E5"}
    mean_fpr = np.linspace(0, 1, 500)
    auc_ci = {}

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=400)
    ax.set_facecolor("white")

    for voi_name in VOI_NAMES:
        folds = all_cv_results[voi_name]
        fold_tprs = []
        for f in folds:
            fpr_f, tpr_f, _ = roc_curve(f["y_true"], f["y_prob"])
            interp = np.interp(mean_fpr, fpr_f, tpr_f); interp[0] = 0.0
            fold_tprs.append(interp)
        fold_tprs = np.array(fold_tprs)
        mean_tpr = fold_tprs.mean(axis=0); mean_tpr[-1] = 1.0
        std_tpr = fold_tprs.std(axis=0)
        mean_auc = np.mean([f["auc"] for f in folds])

        # Bootstrap CI for AUC (2,000 resamples of the pooled out-of-fold predictions)
        all_yt, all_yp = pooled_predictions(folds)
        boot_aucs = []
        for _ in range(2000):
            idx = np.random.choice(len(all_yt), len(all_yt), replace=True)
            if len(np.unique(all_yt[idx])) < 2: continue
            boot_aucs.append(roc_auc_score(all_yt[idx], all_yp[idx]))
        boot_aucs.sort()
        ci_lo, ci_hi = boot_aucs[int(0.025*len(boot_aucs))], boot_aucs[int(0.975*len(boot_aucs))]
        auc_ci[voi_name] = (mean_auc, ci_lo, ci_hi)

        ax.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr, 0), np.minimum(mean_tpr+std_tpr, 1),
                        color=colors[voi_name], alpha=0.12)
        lw = 2.2 if voi_name == "E-VOI" else 2.0
        ax.plot(mean_fpr, mean_tpr, color=colors[voi_name], lw=lw,
                label=f"{voi_name}: Average AUC={mean_auc:.3f} (95% CI: {ci_lo:.3f}–{ci_hi:.3f})")

    ax.plot([0,1],[0,1],"--",color="#AAAAAA",lw=1.0,alpha=0.6,label="Chance (AUC=0.500)")
    ax.set(xlim=[-0.02,1.02], ylim=[-0.02,1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Internal Validation", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig3a_Internal_ROC.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"{output_dir}/Fig3a_Internal_ROC.tiff", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return auc_ci


def clopper_pearson(x, n, alpha=0.05):
    """Exact binomial 95% CI."""
    p = x/n if n > 0 else 0
    lo = stats.beta.ppf(alpha/2, x, n-x+1) if x > 0 else 0.0
    hi = stats.beta.ppf(1-alpha/2, x+1, n-x) if x < n else 1.0
    return p, lo, hi


def summarize_internal(all_cv_results, auc_ci, output_dir):
    """Internal performance: mean AUC of the five folds, sensitivity and specificity
    at the Youden-optimal cutoff of each validation fold (counts pooled over the folds),
    and paired t-tests of the fold AUCs between models."""
    rows = []
    for voi_name in VOI_NAMES:
        tp = fn = tn = fp = 0
        for f in all_cv_results[voi_name]:
            yt, yp = np.array(f["y_true"]), np.array(f["y_prob"])
            fpr, tpr, thr = roc_curve(yt, yp)
            pred = (yp >= thr[np.argmax(tpr - fpr)]).astype(int)
            tp += int(((pred == 1) & (yt == 1)).sum()); fn += int(((pred == 0) & (yt == 1)).sum())
            tn += int(((pred == 0) & (yt == 0)).sum()); fp += int(((pred == 1) & (yt == 0)).sum())
        sens, slo, shi = clopper_pearson(tp, tp+fn)
        spec, plo, phi = clopper_pearson(tn, tn+fp)
        auc, lo, hi = auc_ci[voi_name]
        rows.append({"Model": voi_name, "AUC": f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
                     "Sensitivity": f"{sens:.3f} [{slo:.3f}, {shi:.3f}] ({tp}/{tp+fn})",
                     "Specificity": f"{spec:.3f} [{plo:.3f}, {phi:.3f}] ({tn}/{tn+fp})"})
    perf = pd.DataFrame(rows)
    perf.to_csv(f"{output_dir}/internal_performance.csv", index=False)
    print("\nInternal validation"); print(perf.to_string(index=False))

    comp = []
    for a, b in [("E-VOI", "S-VOI"), ("E-VOI", "R-VOI"), ("R-VOI", "S-VOI")]:
        auc_a = [f["auc"] for f in all_cv_results[a]]; auc_b = [f["auc"] for f in all_cv_results[b]]
        t, p = stats.ttest_rel(auc_a, auc_b)
        comp.append({"Comparison": f"{a} vs {b}", "Mean AUC difference": np.mean(auc_a) - np.mean(auc_b),
                     "t": t, "P": p})
    comp = pd.DataFrame(comp)
    comp.to_csv(f"{output_dir}/internal_auc_comparison.csv", index=False)
    print("\nPaired t-test of fold AUCs"); print(comp.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Internal validation (5-fold cross-validation).")
    ap.add_argument("--data_root", default="./data/train",
                    help="directory that contains R-VOI/, S-VOI/, and E-VOI/")
    ap.add_argument("--labels", default="./data/train/labels.csv")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--reuse_cv", action="store_true",
                    help="skip training and reuse <output_dir>/cv_results.json")
    args = ap.parse_args()

    set_determinism(seed=SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    data_dirs = {v: os.path.join(args.data_root, v) for v in VOI_NAMES}
    cv_path = f"{args.output_dir}/cv_results.json"

    if args.reuse_cv:
        with open(cv_path) as f:
            all_cv_results = json.load(f)
    else:
        labels = load_labels(args.labels)
        all_cv_results = run_cross_validation(data_dirs, labels, args.output_dir, args.num_workers)
        with open(cv_path, "w") as f:
            json.dump(all_cv_results, f, indent=2)
        print(f"\nResults saved to {cv_path}")
        train_final_models(data_dirs, labels, args.output_dir, args.num_workers)

    print("\nCutoffs for external validation (sensitivity >= 80%, maximum specificity)")
    thresholds = determine_thresholds(all_cv_results)
    with open(f"{args.output_dir}/thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    auc_ci = plot_internal_roc(all_cv_results, args.output_dir)
    summarize_internal(all_cv_results, auc_ci, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
