"""External validation of the final DL models and comparison with radiologists.

Steps
  1. Inference: the final models (trained on the entire training set) are applied to
     the external test set with the same preprocessing as in training (no augmentation).
  2. Diagnostic performance: AUC with DeLong 95% CI; sensitivity and specificity with
     exact binomial (Clopper-Pearson) 95% CI at the cutoff predetermined from the
     training set (<model_dir>/thresholds.json) and at a score of >= 3 for the radiologists.
  3. Pairwise DeLong tests (models vs. models, models vs. readers, reader vs. reader).
  4. ROC curves.

Usage
    python external_validation.py --data_root ./data/external \
        --labels ./data/external/labels.csv --readers ./data/external/readers.csv \
        --model_dir ./outputs --output_dir ./outputs/external

To repeat only the statistics from an existing prediction file
(columns: patient_id, label, R-VOI_prob, S-VOI_prob, E-VOI_prob, Reader1, Reader2):
    python external_validation.py --results_csv ./outputs/external/external_predictions.csv \
        --model_dir ./outputs --output_dir ./outputs/external
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOI_NAMES = ["R-VOI", "S-VOI", "E-VOI"]
READER_CUTOFF = 3  # score >= 3 = malignant (5-point scale, 0-4)
BATCH_SIZE = 32


# ── 1. Inference with the final models ──
def predict_external(data_root, labels_csv, model_dir, num_workers):
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.functional import softmax
    from monai.networks.nets import DenseNet121
    from preprocessing import (get_transforms, load_labels, build_file_list,
                               prepare_dict, SafeDataset, collate_fn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    labels = load_labels(labels_csv)
    df = pd.DataFrame({"patient_id": list(labels), "label": list(labels.values())})

    for voi_name in VOI_NAMES:
        print(f"\nInference: {voi_name}")
        items = build_file_list(os.path.join(data_root, voi_name), labels)
        ds = SafeDataset([prepare_dict(it, labels) for it in items], get_transforms(train=False))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn)

        model = DenseNet121(spatial_dims=3, in_channels=3, out_channels=2).to(device)
        model.load_state_dict(torch.load(f"{model_dir}/{voi_name}/final.pth", map_location=device))
        model.eval()

        probs = {}
        with torch.no_grad():
            for batch in loader:
                if batch is None: continue
                imgs = torch.cat([batch["en_img"], batch["t1_img"], batch["t2_img"]], dim=1).to(device)
                p = softmax(model(imgs), dim=1)[:, 1].cpu().tolist()
                probs.update(dict(zip(batch["patient_id"], p)))
        df[f"{voi_name}_prob"] = df["patient_id"].map(probs)

    missing = df[[f"{v}_prob" for v in VOI_NAMES]].isna().any(axis=1)
    if missing.any():
        print(f"Excluded (missing prediction): {df.loc[missing, 'patient_id'].tolist()}")
    return df[~missing].reset_index(drop=True)


# ── 2. Statistical methods ──
def delong_auc_ci(y_true, y_score, alpha=0.05):
    """AUC with DeLong 95% CI."""
    y_true, y_score = np.array(y_true), np.array(y_score)
    pos, neg = y_score[y_true==1], y_score[y_true==0]
    m, n = len(pos), len(neg)
    auc = roc_auc_score(y_true, y_score)
    V10 = np.array([np.mean((neg < p) + 0.5*(neg == p)) for p in pos])
    V01 = np.array([np.mean((pos > q) + 0.5*(pos == q)) for q in neg])
    var = np.var(V10, ddof=1)/m + np.var(V01, ddof=1)/n if (m>1 and n>1) else 0
    z = stats.norm.ppf(1 - alpha/2)
    se = np.sqrt(var)
    return auc, max(0, auc-z*se), min(1, auc+z*se), var


def clopper_pearson(x, n, alpha=0.05):
    """Exact binomial 95% CI."""
    p = x/n if n>0 else 0
    lo = stats.beta.ppf(alpha/2, x, n-x+1) if x>0 else 0.0
    hi = stats.beta.ppf(1-alpha/2, x+1, n-x) if x<n else 1.0
    return p, lo, hi


def delong_test(y_true, y1, y2):
    """Paired DeLong test."""
    a1, _, _, v1 = delong_auc_ci(y_true, y1)
    a2, _, _, v2 = delong_auc_ci(y_true, y2)
    y_true = np.array(y_true); y1, y2 = np.array(y1), np.array(y2)
    pi, ni = np.where(y_true==1)[0], np.where(y_true==0)[0]
    m, n = len(pi), len(ni)
    V10_1 = np.array([np.mean((y1[ni]<y1[i])+0.5*(y1[ni]==y1[i])) for i in pi])
    V10_2 = np.array([np.mean((y2[ni]<y2[i])+0.5*(y2[ni]==y2[i])) for i in pi])
    V01_1 = np.array([np.mean((y1[pi]>y1[j])+0.5*(y1[pi]==y1[j])) for j in ni])
    V01_2 = np.array([np.mean((y2[pi]>y2[j])+0.5*(y2[pi]==y2[j])) for j in ni])
    cov = (np.cov(V10_1,V10_2)[0,1]/m + np.cov(V01_1,V01_2)[0,1]/n) if (m>1 and n>1) else 0
    var_diff = v1 + v2 - 2*cov
    z = (a1-a2)/np.sqrt(var_diff) if var_diff>0 else 0
    return a1, a2, 2*(1-stats.norm.cdf(abs(z)))


# ── 3. Diagnostic performance ──
def diagnostic_performance(df, thresholds, output_dir):
    y_true = df["label"].values
    scores, cutoffs, rows = {}, {}, []
    for name in VOI_NAMES:
        scores[name] = df[f"{name}_prob"].values; cutoffs[name] = thresholds[name]
    for col in ["Reader1", "Reader2"]:
        if col in df.columns:
            scores[col] = df[col].values.astype(float); cutoffs[col] = READER_CUTOFF

    print(f"\n{'Model':<12} {'AUC [95% CI]':<30} {'Sensitivity [95% CI]':<28} {'Specificity [95% CI]':<28}")
    print("-"*98)
    for name, ys in scores.items():
        auc, lo, hi, _ = delong_auc_ci(y_true, ys)
        pred = (ys >= cutoffs[name]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens, slo, shi = clopper_pearson(tp, tp+fn)
        spec, plo, phi = clopper_pearson(tn, tn+fp)
        label = name.replace("Reader", "Reader ")
        print(f"{label:<12} {auc:.3f} [{lo:.3f}, {hi:.3f}]"
              f"          {sens:.3f} [{slo:.3f}, {shi:.3f}]"
              f"        {spec:.3f} [{plo:.3f}, {phi:.3f}]")
        rows.append({"Model": label, "Cutoff": cutoffs[name],
                     "AUC": f"{auc:.3f} [{lo:.3f}, {hi:.3f}]",
                     "Sensitivity": f"{sens:.3f} [{slo:.3f}, {shi:.3f}] ({tp}/{tp+fn})",
                     "Specificity": f"{spec:.3f} [{plo:.3f}, {phi:.3f}] ({tn}/{tn+fp})"})
    pd.DataFrame(rows).to_csv(f"{output_dir}/external_performance.csv", index=False)
    return scores


# ── 4. Pairwise DeLong tests ──
def pairwise_delong(y_true, scores, output_dir):
    pairs = [("E-VOI","S-VOI"),("E-VOI","R-VOI"),("S-VOI","R-VOI"),
             ("E-VOI","Reader1"),("E-VOI","Reader2"),
             ("S-VOI","Reader1"),("S-VOI","Reader2"),
             ("R-VOI","Reader1"),("R-VOI","Reader2"),
             ("Reader1","Reader2")]
    print(f"\n{'Comparison':<25} {'p-value':>9}")
    print("-"*35)
    rows = []
    for n1, n2 in pairs:
        if n1 not in scores or n2 not in scores: continue
        a1, a2, p = delong_test(y_true, scores[n1], scores[n2])
        sig = " *" if p < 0.05 else ""
        print(f"{n1+' vs '+n2:<25} {p:>9.4f}{sig}")
        rows.append({"Comparison": f"{n1} vs {n2}", "AUC 1": a1, "AUC 2": a2, "P": p})
    pd.DataFrame(rows).to_csv(f"{output_dir}/external_delong_tests.csv", index=False)


# ── 5. ROC curves ──
def plot_external_roc(y_true, scores, output_dir):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 11})
    colors = {"R-VOI":"#E53935","S-VOI":"#43A047","E-VOI":"#1E88E5","Reader1":"#7B1FA2","Reader2":"#F57C00"}

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=400)
    ax.set_facecolor("white")

    for name in VOI_NAMES:
        fpr, tpr, _ = roc_curve(y_true, scores[name])
        auc, lo, hi, _ = delong_auc_ci(y_true, scores[name])
        lw = 2.2 if name == "E-VOI" else 2.0
        ax.step(fpr, tpr, where="post", color=colors[name], lw=lw,
                label=f"{name}: AUC={auc:.3f} (95% CI: {lo:.3f}–{hi:.3f})")

    for col, label, ls in [("Reader1","Reader 1","--"),("Reader2","Reader 2","--")]:
        if col not in scores: continue
        fpr, tpr, _ = roc_curve(y_true, scores[col])
        auc, lo, hi, _ = delong_auc_ci(y_true, scores[col])
        ax.step(fpr, tpr, where="post", color=colors[col], lw=1.8, linestyle=ls,
                label=f"{label}: AUC={auc:.3f} (95% CI: {lo:.3f}–{hi:.3f})")

    ax.plot([0,1],[0,1],"--",color="#AAAAAA",lw=1.0,alpha=0.6,label="Chance (AUC=0.500)")
    ax.set(xlim=[-0.02,1.02], ylim=[-0.02,1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("External Validation", fontsize=14, fontweight="bold")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.legend(loc="lower right", fontsize=7.8)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig3b_External_ROC.png", dpi=400, bbox_inches="tight")
    plt.savefig(f"{output_dir}/Fig3b_External_ROC.tiff", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="External validation.")
    ap.add_argument("--data_root", default="./data/external",
                    help="directory that contains R-VOI/, S-VOI/, and E-VOI/")
    ap.add_argument("--labels", default="./data/external/labels.csv")
    ap.add_argument("--readers", default="./data/external/readers.csv",
                    help="CSV with columns patient_id, Reader1, Reader2 (scores 0-4)")
    ap.add_argument("--model_dir", default="./outputs",
                    help="output directory of internal_validation.py (final models and thresholds.json)")
    ap.add_argument("--output_dir", default="./outputs/external")
    ap.add_argument("--results_csv", default=None,
                    help="existing prediction file; if given, inference is skipped")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.results_csv:
        df = pd.read_csv(args.results_csv, dtype={"patient_id": str})
    else:
        df = predict_external(args.data_root, args.labels, args.model_dir, args.num_workers)
        if args.readers and os.path.exists(args.readers):
            readers = pd.read_csv(args.readers, dtype={"patient_id": str})
            df = df.merge(readers[["patient_id", "Reader1", "Reader2"]], on="patient_id", how="left")
        df.to_csv(f"{args.output_dir}/external_predictions.csv", index=False)

    with open(f"{args.model_dir}/thresholds.json") as f:
        thresholds = json.load(f)   # predetermined from the training set

    y_true = df["label"].values
    print(f"n={len(df)}, malignant={int(y_true.sum())}, non-malignant={int((y_true==0).sum())}")
    scores = diagnostic_performance(df, thresholds, args.output_dir)
    pairwise_delong(y_true, scores, args.output_dir)
    plot_external_roc(y_true, scores, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
