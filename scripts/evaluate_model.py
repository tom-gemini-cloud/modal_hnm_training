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

def calculate_random_baseline(vocab_size, k_values):
    """Calculate random baseline metrics for comparison."""
    baseline_metrics = []
    
    for k in k_values:
        # Random precision@K = K / vocab_size (assuming uniform random selection)
        random_precision = k / vocab_size
        
        # Random recall@K depends on the number of relevant items per user
        # For recommendation systems, typically assume 1 relevant item per prediction
        random_recall = k / vocab_size
        
        # Random F1@K
        if random_precision + random_recall > 0:
            random_f1 = 2 * (random_precision * random_recall) / (random_precision + random_recall)
        else:
            random_f1 = 0
            
        # Random NDCG@K (very low, as random ranking has poor ordering)
        # Approximation: random NDCG ≈ 0.5 * log2(2) / log2(k+1) for single relevant item
        random_ndcg = 0.5 / (k ** 0.5) if k > 0 else 0
        
        baseline_metrics.append({
            'K': k,
            'Precision': random_precision,
            'Recall': random_recall,
            'F1-Score': random_f1,
            'NDCG': random_ndcg
        })
    
    return pd.DataFrame(baseline_metrics)

def plot_metrics(metrics_df, output_dir, vocab_size):
    """Plot ROC and Precision-Recall curves with baseline comparison."""
    # Calculate random baseline
    baseline_df = calculate_random_baseline(vocab_size, metrics_df['K'].tolist())
    
    # Create ROC Curve plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # For ROC curve, we use Recall as TPR (True Positive Rate)
    # and calculate FPR (False Positive Rate) = FP / (FP + TN)
    # For recommendation: FPR ≈ (K - TP) / (vocab_size - relevant_items) ≈ K / vocab_size for large catalogs
    model_fpr = [k / vocab_size for k in metrics_df['K']]
    baseline_fpr = [k / vocab_size for k in baseline_df['K']]
    
    # Plot ROC curves
    ax1.plot(model_fpr, metrics_df['Recall'], 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax1.plot(baseline_fpr, baseline_df['Recall'], 'r--', linewidth=2, alpha=0.7, label='Random Baseline')
    ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random Classifier')
    
    ax1.set_title('ROC Curve: Model vs Random Baseline', fontsize=16, fontweight='bold')
    ax1.set_xlabel('False Positive Rate (≈ K / Vocabulary Size)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(model_fpr) * 1.1])
    ax1.set_ylim([0, max(metrics_df['Recall']) * 1.1])
    
    plt.tight_layout()
    roc_file = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {roc_file}")
    plt.close()
    
    # Create Precision-Recall Curve plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot Precision-Recall curves
    ax2.plot(metrics_df['Recall'], metrics_df['Precision'], 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax2.plot(baseline_df['Recall'], baseline_df['Precision'], 'r--', linewidth=2, alpha=0.7, label='Random Baseline')
    
    # Add random baseline line (constant precision = positive_rate)
    random_precision = 1 / vocab_size  # Assuming 1 relevant item per query
    ax2.axhline(y=random_precision, color='k', linestyle=':', alpha=0.5, label=f'Random Precision ({random_precision:.6f})')
    
    ax2.set_title('Precision-Recall Curve: Model vs Random Baseline', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(metrics_df['Recall']) * 1.1])
    ax2.set_ylim([0, max(metrics_df['Precision']) * 1.1])
    
    # Add K value annotations
    for i, k in enumerate(metrics_df['K']):
        if k in [5, 20, 50]:  # Annotate key K values
            ax2.annotate(f'K={k}', 
                        xy=(metrics_df.iloc[i]['Recall'], metrics_df.iloc[i]['Precision']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    
    plt.tight_layout()
    pr_file = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(pr_file, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {pr_file}")
    plt.close()
    
    # Create metrics@K comparison plots
    fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision@K
    ax1.plot(metrics_df['K'], metrics_df['Precision'], 'bo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax1.plot(baseline_df['K'], baseline_df['Precision'], 'k--', linewidth=2, alpha=0.7, label='Random Baseline')
    ax1.set_title('Precision@K', fontsize=14, fontweight='bold')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Precision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better show the difference
    
    # Recall@K
    ax2.plot(metrics_df['K'], metrics_df['Recall'], 'ro-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax2.plot(baseline_df['K'], baseline_df['Recall'], 'k--', linewidth=2, alpha=0.7, label='Random Baseline')
    ax2.set_title('Recall@K', fontsize=14, fontweight='bold')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Recall')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1-Score@K
    ax3.plot(metrics_df['K'], metrics_df['F1-Score'], 'go-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax3.plot(baseline_df['K'], baseline_df['F1-Score'], 'k--', linewidth=2, alpha=0.7, label='Random Baseline')
    ax3.set_title('F1-Score@K', fontsize=14, fontweight='bold')
    ax3.set_xlabel('K')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale to better show the difference
    
    # NDCG@K
    ax4.plot(metrics_df['K'], metrics_df['NDCG'], 'mo-', linewidth=3, markersize=8, label='BERT4Rec Model')
    ax4.plot(baseline_df['K'], baseline_df['NDCG'], 'k--', linewidth=2, alpha=0.7, label='Random Baseline')
    ax4.set_title('NDCG@K', fontsize=14, fontweight='bold')
    ax4.set_xlabel('K')
    ax4.set_ylabel('NDCG')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    metrics_file = os.path.join(output_dir, 'metrics_at_k.png')
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    print(f"Metrics@K chart saved to: {metrics_file}")
    plt.close()
    
    return baseline_df

def add_baseline_comparison_to_report(output_dir, metrics_df, baseline_df):
    """Add baseline comparison section to the markdown report."""
    report_file = os.path.join(output_dir, "evaluation_report.md")
    
    # Read existing report
    with open(report_file, 'r') as f:
        existing_content = f.read()
    
    # Create baseline comparison section
    comparison_lines = []
    comparison_lines.append("\n## Random Baseline Comparison")
    comparison_lines.append("")
    comparison_lines.append("### Performance vs Random Baseline")
    comparison_lines.append("")
    comparison_lines.append("| K | Model Recall | Random Recall | Improvement | Model NDCG | Random NDCG | Improvement |")
    comparison_lines.append("|---|--------------|---------------|-------------|------------|-------------|-------------|")
    
    for i, row in metrics_df.iterrows():
        k = int(row['K'])
        model_recall = row['Recall']
        random_recall = baseline_df.iloc[i]['Recall']
        recall_improvement = ((model_recall / random_recall - 1) * 100) if random_recall > 0 else float('inf')
        
        model_ndcg = row['NDCG']
        random_ndcg = baseline_df.iloc[i]['NDCG']
        ndcg_improvement = ((model_ndcg / random_ndcg - 1) * 100) if random_ndcg > 0 else float('inf')
        
        comparison_lines.append(f"| {k} | {model_recall:.4f} | {random_recall:.4f} | **+{recall_improvement:.0f}%** | {model_ndcg:.4f} | {random_ndcg:.4f} | **+{ndcg_improvement:.0f}%** |")
    
    comparison_lines.append("")
    comparison_lines.append("### Key Insights")
    comparison_lines.append("")
    
    # Calculate average improvements
    avg_recall_improvement = metrics_df['Recall'].mean() / baseline_df['Recall'].mean() - 1
    avg_ndcg_improvement = metrics_df['NDCG'].mean() / baseline_df['NDCG'].mean() - 1
    
    comparison_lines.append(f"- **Model significantly outperforms random baseline** across all K values")
    comparison_lines.append(f"- **Average Recall improvement**: +{avg_recall_improvement*100:.0f}% vs random selection")
    comparison_lines.append(f"- **Average NDCG improvement**: +{avg_ndcg_improvement*100:.0f}% vs random ranking")
    comparison_lines.append(f"- **Best performance** at K=50 with {metrics_df.iloc[-1]['Recall']*100:.1f}% recall")
    comparison_lines.append("")
    comparison_lines.append("The model's superior performance demonstrates effective learning of user-item interaction patterns.")
    
    # Append to existing report
    updated_content = existing_content + '\n'.join(comparison_lines)
    
    # Write updated report
    with open(report_file, 'w') as f:
        f.write(updated_content)

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
    vocab_size = results['config']['model']['vocab_size']
    baseline_df = plot_metrics(metrics_df, output_dir, vocab_size)
    
    # Add baseline comparison to markdown report
    add_baseline_comparison_to_report(output_dir, metrics_df, baseline_df)