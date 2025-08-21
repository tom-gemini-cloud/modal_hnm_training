import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from math import log2

def load_and_analyse_results(results_path, output_dir):
    """Load and analyse model evaluation results."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    test_metrics = results['test_metrics']

    # Extract metrics by category
    accuracy_metrics = {k: v for k, v in test_metrics.items() if 'accuracy' in k}
    precision_metrics = {k: v for k, v in test_metrics.items() if 'precision' in k}
    recall_metrics = {k: v for k, v in test_metrics.items() if 'recall' in k}
    f1_metrics = {k: v for k, v in test_metrics.items() if 'f1' in k}
    ndcg_metrics = {k: v for k, v in test_metrics.items() if 'ndcg' in k}

    # Create Markdown output report
    report_lines = []
    report_lines.append("# Model Evaluation Results")
    report_lines.append("")
    report_lines.append(f"**Experiment:** {results['experiment_name']}")
    report_lines.append(f"**Best Epoch:** {results['best_epoch']}")
    report_lines.append(f"**MLM Loss:** {test_metrics['mlm_loss']:.4f}")
    report_lines.append("")

    report_lines.append("## Accuracy Metrics")
    for metric, value in accuracy_metrics.items():
        report_lines.append(f"- **{metric}:** {value:.4f} ({value*100:.2f}%)")
    report_lines.append("")

    report_lines.append("## Precision@K")
    for metric, value in precision_metrics.items():
        k = metric.split('@')[1]
        report_lines.append(f"- **P@{k}:** {value:.4f} ({value*100:.2f}%)")
    report_lines.append("")

    report_lines.append("## Recall@K")
    for metric, value in recall_metrics.items():
        k = metric.split('@')[1]
        report_lines.append(f"- **R@{k}:** {value:.4f} ({value*100:.2f}%)")
    report_lines.append("")

    report_lines.append("## F1-Score@K")
    for metric, value in f1_metrics.items():
        k = metric.split('@')[1]
        report_lines.append(f"- **F1@{k}:** {value:.4f} ({value*100:.2f}%)")
    report_lines.append("")

    report_lines.append("## NDCG@K")
    for metric, value in ndcg_metrics.items():
        k = metric.split('@')[1]
        report_lines.append(f"- **NDCG@{k}:** {value:.4f}")
    report_lines.append("")

    # Create summary DataFrame
    k_values = [5, 10, 20, 50]
    metrics_df = pd.DataFrame({
        'K': k_values,
        'Precision': [precision_metrics[f'precision@{k}'] for k in k_values],
        'Recall': [recall_metrics[f'recall@{k}'] for k in k_values],
        'F1-Score': [f1_metrics[f'f1@{k}'] for k in k_values],
        'NDCG': [ndcg_metrics[f'ndcg@{k}'] for k in k_values]
    })

    report_lines.append("## Summary Table")
    report_lines.append("")
    report_lines.append("| K | Precision | Recall | F1-Score | NDCG |")
    report_lines.append("|---|-----------|--------|----------|------|")
    for _, row in metrics_df.iterrows():
        report_lines.append(f"| {int(row['K'])} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['NDCG']:.4f} |")
    report_lines.append("")

    # Model configuration section
    report_lines.append("## Model Configuration")
    report_lines.append("")
    config = results['config']
    report_lines.append("### Model Parameters")
    for key, value in config['model'].items():
        report_lines.append(f"- **{key}:** {value}")
    report_lines.append("")
    report_lines.append("### Training Parameters")
    for key, value in config['training'].items():
        report_lines.append(f"- **{key}:** {value}")
    report_lines.append("")
    report_lines.append("### Data Parameters")
    for key, value in config['data'].items():
        report_lines.append(f"- **{key}:** {value}")

    # Print concise console summary
    print("=== MODEL EVALUATION RESULTS ===")
    print(f"Experiment: {results['experiment_name']}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"MLM Loss: {test_metrics['mlm_loss']:.4f}")
    print("\nSUMMARY TABLE:")
    print(metrics_df.round(4))

    # Save Markdown report and CSV
    report_file = os.path.join(output_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    csv_file = os.path.join(output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)

    return results, metrics_df

def _discount_sum(k: int) -> float:
    """Sum of discounts 1/log2(i+1) for i=1..k."""
    return sum(1.0 / log2(i + 1) for i in range(1, k + 1)) if k > 0 else 0.0

def calculate_random_baseline(vocab_size, k_values):
    """
    Calculate random baseline metrics for comparison under a uniform
    top-K guess over a catalogue of size |V| = vocab_size and one
    relevant item per query.

    Precision_rand = 1/|V|  (constant)
    Recall_rand    = K/|V|
    F1_rand        = 2 * P * R / (P + R) = 2K / (|V|(K+1))
    NDCG_rand      = E[DCG@K] / IDCG = (sum_{i=1}^K 1/log2(i+1)) / |V|
    """
    baseline_metrics = []
    p_rand = 1.0 / float(vocab_size)
    for k in k_values:
        r_rand = float(k) / float(vocab_size)
        f1_rand = (2 * p_rand * r_rand) / (p_rand + r_rand) if (p_rand + r_rand) > 0 else 0.0
        ndcg_rand = _discount_sum(k) / float(vocab_size)
        baseline_metrics.append({
            'K': int(k),
            'Precision': p_rand,
            'Recall': r_rand,
            'F1-Score': f1_rand,
            'NDCG': ndcg_rand
        })
    return pd.DataFrame(baseline_metrics)

def plot_metrics(metrics_df, output_dir, vocab_size):
    """Plot ROC and Precision–Recall curves with corrected random baselines."""
    baseline_df = calculate_random_baseline(vocab_size, metrics_df['K'].tolist())

    # --- ROC Curve (TPR = Recall, FPR ≈ K / |catalogue|) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    fpr = [k / vocab_size for k in metrics_df['K']]
    tpr = metrics_df['Recall'].values

    ax.plot(fpr, tpr, 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')

    # only the 45° random diagonal
    xmax = max(fpr) * 1.1
    ax.plot([0, xmax], [0, xmax], ':', color='0.2', label='Random Classifier')

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max(tpr) * 1.1)
    ax.set_title('ROC Curve: Model vs Random', fontsize=16, fontweight='bold')
    ax.set_xlabel('False Positive Rate (≈ K / Catalogue Size)')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Precision–Recall Curve ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

    ax2.plot(metrics_df['Recall'], metrics_df['Precision'], 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax2.plot(baseline_df['Recall'], baseline_df['Precision'], 'r--', linewidth=2, alpha=0.8, label='Random Baseline')

    # Random precision is constant at 1/|V|
    random_precision = 1.0 / float(vocab_size)
    ax2.axhline(y=random_precision, color='k', linestyle=':', alpha=0.6,
                label=f'Random Precision ({random_precision:.6f})')

    ax2.set_title('Precision–Recall Curve: Model vs Random Baseline', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(metrics_df['Recall']) * 1.1])
    ax2.set_ylim([0, max(metrics_df['Precision']) * 1.1])

    # Annotate all K points for clarity
    for i, k in enumerate(metrics_df['K']):
        ax2.annotate(f'K={int(k)}',
                     xy=(metrics_df.iloc[i]['Recall'], metrics_df.iloc[i]['Precision']),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.85)

    plt.tight_layout()
    pr_file = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_file, dpi=300, bbox_inches='tight')
    print(f"Precision–Recall curve saved to: {pr_file}")
    plt.close()

    # --- Metrics@K panel ---
    fig3, ((axp, axr), (axf, axn)) = plt.subplots(2, 2, figsize=(14, 10))

    # Precision@K
    axp.plot(metrics_df['K'], metrics_df['Precision'], 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    axp.plot(baseline_df['K'], baseline_df['Precision'], 'k--', linewidth=2, alpha=0.8, label='Random Baseline')
    axp.set_title('Precision@K', fontsize=14, fontweight='bold')
    axp.set_xlabel('K'); axp.set_ylabel('Precision')
    axp.legend(); axp.grid(True, alpha=0.3); axp.set_yscale('log')

    # Recall@K
    axr.plot(metrics_df['K'], metrics_df['Recall'], 'ro-', linewidth=3, markersize=8, label='BERT4Rec Model')
    axr.plot(baseline_df['K'], baseline_df['Recall'], 'k--', linewidth=2, alpha=0.8, label='Random Baseline')
    axr.set_title('Recall@K', fontsize=14, fontweight='bold')
    axr.set_xlabel('K'); axr.set_ylabel('Recall')
    axr.legend(); axr.grid(True, alpha=0.3)

    # F1-Score@K
    axf.plot(metrics_df['K'], metrics_df['F1-Score'], 'go-', linewidth=3, markersize=8, label='BERT4Rec Model')
    axf.plot(baseline_df['K'], baseline_df['F1-Score'], 'k--', linewidth=2, alpha=0.8, label='Random Baseline')
    axf.set_title('F1-Score@K', fontsize=14, fontweight='bold')
    axf.set_xlabel('K'); axf.set_ylabel('F1-Score')
    axf.legend(); axf.grid(True, alpha=0.3); axf.set_yscale('log')

    # NDCG@K
    axn.plot(metrics_df['K'], metrics_df['NDCG'], 'mo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    axn.plot(baseline_df['K'], baseline_df['NDCG'], 'k--', linewidth=2, alpha=0.8, label='Random Baseline')
    axn.set_title('NDCG@K', fontsize=14, fontweight='bold')
    axn.set_xlabel('K'); axn.set_ylabel('NDCG')
    axn.legend(); axn.grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_file = os.path.join(output_dir, 'metrics_at_k.png')
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"Metrics@K chart saved to: {metrics_file}")
    plt.close()

    return baseline_df

def add_baseline_comparison_to_report(output_dir, metrics_df, baseline_df):
    """Append a baseline comparison section to the Markdown report."""
    report_file = os.path.join(output_dir, "evaluation_report.md")
    with open(report_file, 'r') as f:
        existing_content = f.read()

    def fmt_factor(m, b):
        if b <= 0:
            return "—"
        factor = m / b
        return f"{factor:,.1f}×" if factor < 10000 else f"{factor:,.0f}×"

    lines = []
    lines.append("\n## Random Baseline Comparison\n")
    lines.append("### Performance vs Random Baseline\n")
    lines.append("| K | Model Recall | Random Recall | Recall (× random) | Model NDCG | Random NDCG | NDCG (× random) |")
    lines.append("|---|--------------|---------------|-------------------|------------|-------------|-----------------|")

    for i, row in metrics_df.iterrows():
        k = int(row['K'])
        mr, br = row['Recall'], baseline_df.iloc[i]['Recall']
        mn, bn = row['NDCG'], baseline_df.iloc[i]['NDCG']
        lines.append(f"| {k} | {mr:.4f} | {br:.6f} | {fmt_factor(mr, br)} | {mn:.4f} | {bn:.6f} | {fmt_factor(mn, bn)} |")

    lines.append("")
    lines.append("### Key Insights\n")
    lines.append("- The model outperforms a uniform random selector by large margins at all K values.")
    lines.append("- Random precision is constant at 1/|V|; model precision declines with K while recall increases, reflecting the usual top-K trade-off.")
    lines.append("- Random NDCG is near zero because it is the unconditional expectation over the full catalogue.")

    updated_content = existing_content + "\n".join(lines)
    with open(report_file, 'w') as f:
        f.write(updated_content)

if __name__ == "__main__":
    # Directory setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Create evaluation output directory in project root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, f"evaluation_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created evaluation output directory: {output_dir}")

    # Path to results file relative to project root
    results_path = os.path.join(project_root, "data", "bert4rec_output", "bert4rec_20250821_121512", "results.json")
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        print("Available result directories:")
        results_base = os.path.join(project_root, "data", "bert4rec_output")
        if os.path.exists(results_base):
            for item in os.listdir(results_base):
                print(f"  - {item}")
        sys.exit(1)

    results, metrics_df = load_and_analyse_results(results_path, output_dir)
    vocab_size = results['config']['model']['vocab_size']
    baseline_df = plot_metrics(metrics_df, output_dir, vocab_size)
    add_baseline_comparison_to_report(output_dir, metrics_df, baseline_df)
