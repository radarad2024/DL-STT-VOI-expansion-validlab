#!/usr/bin/env python3
"""
External Validation Script
- Apply training-derived optimal thresholds
- DeLong method for AUC confidence intervals
- DeLong test for model comparisons
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt

# Set font to Arial
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Configuration for external validation"""
    # Paths (MODIFY THESE)
    predictions_csv = "/path/to/external_predictions.csv"
    optimal_thresholds_json = "./outputs/5cv_results/optimal_thresholds.json"
    output_dir = "./outputs/external_validation"

    # Column names in predictions CSV
    # Expected columns: 'True_Label', 'BoundingBox_Prob', 'StandardVOI_Prob', 'ExpandedVOI_Prob'
    label_column = 'True_Label'
    model_prob_columns = {
        'Rectangular VOI': 'BoundingBox_Prob',
        'StandardVOI': 'StandardVOI_Prob',
        'ExpandedVOI': 'ExpandedVOI_Prob'
    }


# ============================================================================
# DeLong Method
# ============================================================================
def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance using DeLong method

    Args:
        ground_truth: True binary labels (numpy array)
        predictions: Predicted probabilities (numpy array)

    Returns:
        auc_value: AUC score
        variance: Variance of AUC
    """
    # Separate positive and negative examples
    positive_examples = predictions[ground_truth == 1]
    negative_examples = predictions[ground_truth == 0]

    m = len(positive_examples)  # Number of positive samples
    n = len(negative_examples)  # Number of negative samples

    # Compute AUC
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    auc_value = auc(fpr, tpr)

    # Compute structural components for positive class
    V10 = np.zeros(m)
    for i, pos in enumerate(positive_examples):
        V10[i] = (np.sum(negative_examples < pos) +
                  0.5 * np.sum(negative_examples == pos)) / n

    # Compute structural components for negative class
    V01 = np.zeros(n)
    for i, neg in enumerate(negative_examples):
        V01[i] = (np.sum(positive_examples > neg) +
                  0.5 * np.sum(positive_examples == neg)) / m

    # Variance components
    S10 = np.var(V10, ddof=1) if m > 1 else 0
    S01 = np.var(V01, ddof=1) if n > 1 else 0

    # Total variance
    var_auc = (S10 / m) + (S01 / n)

    return auc_value, var_auc


def delong_ci(ground_truth, predictions, alpha=0.05):
    """
    Compute DeLong 95% confidence interval for AUC

    Args:
        ground_truth: True binary labels
        predictions: Predicted probabilities
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        auc_value: AUC score
        ci_lower: Lower bound of CI
        ci_upper: Upper bound of CI
        se: Standard error
    """
    auc_value, var_auc = delong_roc_variance(ground_truth, predictions)
    se = np.sqrt(var_auc)

    # Calculate CI using normal distribution
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = auc_value - z * se
    ci_upper = auc_value + z * se

    return auc_value, ci_lower, ci_upper, se


def delong_test(ground_truth, predictions_1, predictions_2):
    """
    DeLong test for comparing two correlated ROC curves

    Args:
        ground_truth: True binary labels
        predictions_1: Predictions from model 1
        predictions_2: Predictions from model 2

    Returns:
        z_statistic: Test statistic
        p_value: Two-tailed p-value
    """
    # Calculate AUC and variance for both models
    auc1, var1 = delong_roc_variance(ground_truth, predictions_1)
    auc2, var2 = delong_roc_variance(ground_truth, predictions_2)

    # Separate positive and negative examples
    positive_examples_1 = predictions_1[ground_truth == 1]
    negative_examples_1 = predictions_1[ground_truth == 0]

    positive_examples_2 = predictions_2[ground_truth == 1]
    negative_examples_2 = predictions_2[ground_truth == 0]

    m = len(positive_examples_1)
    n = len(negative_examples_1)

    # Compute structural components for both models
    V10_1 = np.zeros(m)
    V10_2 = np.zeros(m)

    for i in range(m):
        pos1 = positive_examples_1[i]
        pos2 = positive_examples_2[i]

        V10_1[i] = (np.sum(negative_examples_1 < pos1) +
                    0.5 * np.sum(negative_examples_1 == pos1)) / n
        V10_2[i] = (np.sum(negative_examples_2 < pos2) +
                    0.5 * np.sum(negative_examples_2 == pos2)) / n

    # Covariance between the two models
    cov_01 = np.cov(V10_1, V10_2)[0, 1] / m if m > 1 else 0

    # Variance of the difference
    var_diff = var1 + var2 - 2 * cov_01
    se_diff = np.sqrt(var_diff)

    # Test statistic
    z_stat = (auc1 - auc2) / se_diff if se_diff > 0 else 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


# ============================================================================
# Metrics Calculation
# ============================================================================
def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    """
    Calculate performance metrics at a specific threshold

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'threshold': float(threshold),
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
# External Validation
# ============================================================================
def evaluate_external_validation(predictions_csv, optimal_thresholds_json, output_dir):
    """
    Evaluate external validation using training optimal thresholds

    Args:
        predictions_csv: Path to external predictions CSV
        optimal_thresholds_json: Path to optimal thresholds from training
        output_dir: Output directory
    """
    print("="*80)
    print("EXTERNAL VALIDATION EVALUATION")
    print("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    df = pd.read_csv(predictions_csv)
    print(f"\n✅ Loaded {len(df)} samples from external test set")
    print(f"   Positive samples: {df[Config.label_column].sum()}")
    print(f"   Negative samples: {len(df) - df[Config.label_column].sum()}")

    # Load optimal thresholds from training
    with open(optimal_thresholds_json, 'r') as f:
        optimal_thresholds = json.load(f)

    print(f"\n✅ Loaded optimal thresholds from training:")
    for model_name, threshold in optimal_thresholds.items():
        print(f"   {model_name}: {threshold:.4f}")

    # Evaluate each model
    results = {}
    y_true = df[Config.label_column].values

    for model_name, prob_col in Config.model_prob_columns.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        y_prob = df[prob_col].values

        # Calculate AUC with DeLong CI
        auc_value, ci_lower, ci_upper, se = delong_ci(y_true, y_prob)

        print(f"AUC: {auc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        print(f"Standard Error: {se:.4f}")

        # Get training optimal threshold
        training_threshold = optimal_thresholds[model_name]
        print(f"\nApplying training optimal threshold: {training_threshold:.4f}")

        # Calculate metrics at training threshold
        metrics = calculate_metrics_at_threshold(y_true, y_prob, training_threshold)

        print(f"Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"Specificity: {metrics['specificity']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"PPV: {metrics['ppv']:.3f}")
        print(f"NPV: {metrics['npv']:.3f}")
        print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, "
              f"FP={metrics['fp']}, FN={metrics['fn']}")

        # Store results
        results[model_name] = {
            'auc': auc_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'se': se,
            'ci_method': 'DeLong',
            'training_threshold': training_threshold,
            'metrics': metrics,
            'y_true': y_true.tolist(),
            'y_prob': y_prob.tolist()
        }

    # Perform DeLong tests for pairwise comparisons
    print(f"\n{'='*80}")
    print("DELONG TEST FOR MODEL COMPARISONS")
    print(f"{'='*80}")

    delong_results = {}
    model_list = list(Config.model_prob_columns.keys())

    for i in range(len(model_list)):
        for j in range(i + 1, len(model_list)):
            model1, model2 = model_list[i], model_list[j]

            prob1 = df[Config.model_prob_columns[model1]].values
            prob2 = df[Config.model_prob_columns[model2]].values

            z_stat, p_value = delong_test(y_true, prob1, prob2)

            comparison_key = f"{model1} vs {model2}"
            delong_results[comparison_key] = {
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'auc_diff': float(results[model1]['auc'] - results[model2]['auc'])
            }

            print(f"\n{comparison_key}:")
            print(f"  AUC difference: {delong_results[comparison_key]['auc_diff']:.4f}")
            print(f"  Z-statistic: {z_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

    # Save detailed results
    detailed_results = {
        'n_samples': len(df),
        'n_positive': int(df[Config.label_column].sum()),
        'n_negative': int(len(df) - df[Config.label_column].sum()),
        'models': results,
        'delong_tests': delong_results
    }

    detailed_path = output_dir / 'external_validation_results.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=float)

    print(f"\n✅ Detailed results saved: {detailed_path}")

    # Create summary tables
    create_summary_tables(results, delong_results, output_dir)

    # Generate ROC curves
    plot_roc_curves(results, output_dir)

    return results, delong_results


def create_summary_tables(results, delong_results, output_dir):
    """
    Create summary tables in multiple formats

    Args:
        results: Dictionary of evaluation results
        delong_results: DeLong test results
        output_dir: Output directory
    """
    # Prepare data for table
    table_data = []
    for model_name, res in results.items():
        metrics = res['metrics']
        table_data.append({
            'Model': model_name,
            'AUC (95% CI)': f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})",
            'Threshold*': f"{res['training_threshold']:.3f}",
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'PPV': f"{metrics['ppv']:.3f}",
            'NPV': f"{metrics['npv']:.3f}"
        })

    df = pd.DataFrame(table_data)

    # Save CSV
    csv_path = output_dir / 'external_validation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV summary saved: {csv_path}")

    # Save markdown
    md_path = output_dir / 'external_validation_summary.md'
    with open(md_path, 'w') as f:
        f.write("# External Validation Results\n\n")
        f.write("## Performance Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*Threshold from training set (mean optimal threshold across 5 folds)\n\n")

        f.write("## DeLong Test Results\n\n")
        f.write("| Comparison | AUC Difference | Z-statistic | P-value | Significant (p<0.05) |\n")
        f.write("|------------|----------------|-------------|---------|---------------------|\n")
        for comp, stats in delong_results.items():
            sig_str = "Yes" if stats['significant'] else "No"
            f.write(f"| {comp} | {stats['auc_diff']:.4f} | "
                   f"{stats['z_statistic']:.4f} | {stats['p_value']:.4f} | {sig_str} |\n")

        f.write("\n## Notes\n\n")
        f.write("- **AUC 95% CI**: Calculated using DeLong method\n")
        f.write("- **Threshold**: Optimal threshold from training (NOT calculated from external data)\n")
        f.write("- **DeLong test**: Statistical test for comparing correlated ROC curves\n")
        f.write("- **Significance level**: p < 0.05\n")

    print(f"✅ Markdown summary saved: {md_path}")


def plot_roc_curves(results, output_dir):
    """
    Plot ROC curves for all models

    Args:
        results: Dictionary of evaluation results
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        'Rectangular VOI': '#d62728',
        'StandardVOI': '#2ca02c',
        'ExpandedVOI': '#1f77b4'
    }

    for model_name, res in results.items():
        y_true = np.array(res['y_true'])
        y_prob = np.array(res['y_prob'])

        fpr, tpr, _ = roc_curve(y_true, y_prob)

        color = colors.get(model_name, '#333333')

        ax.plot(fpr, tpr, color=color, lw=2.5,
               label=f'{model_name}: AUC={res["auc"]:.3f} '
                     f'(95% CI: {res["ci_lower"]:.3f}-{res["ci_upper"]:.3f})')

    # Chance line
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
           label='Chance (AUC=0.500)', alpha=0.7)

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves - External Validation', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, labelsize=11)

    plt.tight_layout()

    # Save
    png_path = output_dir / 'roc_curves.png'
    pdf_path = output_dir / 'roc_curves.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC curves saved: {png_path}, {pdf_path}")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXTERNAL VALIDATION WITH DELONG METHOD")
    print("="*80)
    print(f"Predictions: {Config.predictions_csv}")
    print(f"Thresholds: {Config.optimal_thresholds_json}")
    print(f"Output: {Config.output_dir}")
    print("="*80 + "\n")

    # Run evaluation
    results, delong_tests = evaluate_external_validation(
        Config.predictions_csv,
        Config.optimal_thresholds_json,
        Config.output_dir
    )

    print("\n" + "="*80)
    print("EXTERNAL VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {Config.output_dir}")

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for model_name, res in results.items():
        print(f"\n{model_name}:")
        print(f"  AUC: {res['auc']:.3f} (95% CI: {res['ci_lower']:.3f}-{res['ci_upper']:.3f})")
        print(f"  Sensitivity: {res['metrics']['sensitivity']:.3f}")
        print(f"  Specificity: {res['metrics']['specificity']:.3f}")


if __name__ == "__main__":
    main()
