import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

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
        report_lines.append(f"- **NDCG@{k}:** {value:.4f} ({value*100:.2f}%)")
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
    # Create markdown table
    report_lines.append("| K | Precision | Recall | F1-Score | NDCG |")
    report_lines.append("|---|-----------|--------|----------|------|")
    for _, row in metrics_df.iterrows():
        report_lines.append(f"| {int(row['K'])} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['NDCG']:.4f} |")
    report_lines.append("")
    
    # Add model configuration section
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
    
    # Print to console (simplified version)
    console_lines = []
    console_lines.append("=== MODEL EVALUATION RESULTS ===")
    console_lines.append(f"Experiment: {results['experiment_name']}")
    console_lines.append(f"Best Epoch: {results['best_epoch']}")
    console_lines.append(f"MLM Loss: {test_metrics['mlm_loss']:.4f}")
    console_lines.append("")
    console_lines.append("SUMMARY TABLE:")
    console_lines.append(str(metrics_df.round(4)))
    
    for line in console_lines:
        print(line)
    
    # Save Markdown report to file
    report_file = os.path.join(output_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save metrics as CSV
    csv_file = os.path.join(output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(csv_file, index=False)
    
    return results, metrics_df

def plot_metrics(metrics_df, output_dir):
    """Plot metrics visualisation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Precision@K
    ax1.plot(metrics_df['K'], metrics_df['Precision'], 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Precision@K', fontsize=14, fontweight='bold')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Precision')
    ax1.grid(True, alpha=0.3)
    
    # Recall@K
    ax2.plot(metrics_df['K'], metrics_df['Recall'], 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Recall@K', fontsize=14, fontweight='bold')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Recall')
    ax2.grid(True, alpha=0.3)
    
    # F1-Score@K
    ax3.plot(metrics_df['K'], metrics_df['F1-Score'], 'go-', linewidth=2, markersize=8)
    ax3.set_title('F1-Score@K', fontsize=14, fontweight='bold')
    ax3.set_xlabel('K')
    ax3.set_ylabel('F1-Score')
    ax3.grid(True, alpha=0.3)
    
    # NDCG@K
    ax4.plot(metrics_df['K'], metrics_df['NDCG'], 'mo-', linewidth=2, markersize=8)
    ax4.set_title('NDCG@K', fontsize=14, fontweight='bold')
    ax4.set_xlabel('K')
    ax4.set_ylabel('NDCG')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to output directory
    plot_file = os.path.join(output_dir, 'model_evaluation_metrics.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    plt.show()

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level to project root
    
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
    plot_metrics(metrics_df, output_dir)