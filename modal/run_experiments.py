# modal/run_experiments.py
"""
Run multiple BERT4Rec experiments with different hyperparameters on Modal.
This script allows you to define and run multiple training configurations.
"""

from modal import App, gpu
from datetime import datetime
import json
from typing import Dict, List, Any

# Import the training app
from modal.train_modal import app, train_bert4rec_job

class ExperimentRunner:
    """Manages and runs multiple BERT4Rec experiments on Modal."""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        """
        Initialise the experiment runner with a base configuration.
        
        Args:
            base_config: Base configuration that all experiments will inherit
        """
        self.base_config = base_config or self.get_default_config()
        self.results = []
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get the default configuration for BERT4Rec training."""
        return {
            # Data parameters
            "train_glob": "/data/modelling_data/*.parquet",
            
            # Model architecture
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 3,
            "dim_feedforward": 512,
            "max_len": 100,
            "dropout": 0.1,
            
            # Training parameters
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "n_epochs": 10,
            "warmup_steps": 1000,
            "grad_clip_norm": 1.0,
            
            # Sequence parameters
            "min_seq_len": 3,
            "deduplicate_exact": True,
            "treat_same_day_as_basket": True,
            
            # Prefix tokens
            "add_segment_prefix": False,
            "add_channel_prefix": False,
            "add_priceband_prefix": False,
            "n_price_bins": 10,
            
            # Masking parameters
            "mask_prob": 0.15,
            "random_token_prob": 0.10,
            "keep_original_prob": 0.10,
            
            # Split ratios (80:10:10)
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "test_ratio": 0.1,
            
            # Output
            "out_dir": "/data/bert4rec_output",
            
            # Evaluation
            "eval_topk": 20,
            "eval_every_n_epochs": 2,
            
            # Random seed
            "seed": 42,
        }
    
    def create_experiment(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create an experiment configuration by overriding base config.
        
        Args:
            name: Experiment name
            **kwargs: Parameters to override in the base configuration
        
        Returns:
            Complete experiment configuration
        """
        config = self.base_config.copy()
        config.update(kwargs)
        config["experiment_name"] = name
        return config
    
    def run_experiment(self, config: Dict[str, Any], gpu_type: str = "L4") -> Dict[str, Any]:
        """
        Run a single experiment on Modal.
        
        Args:
            config: Experiment configuration
            gpu_type: GPU type to use (L4, A10G, or A100)
        
        Returns:
            Experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {config['experiment_name']}")
        print(f"{'='*60}")
        
        # Map GPU type to Modal GPU object
        gpu_map = {
            "L4": gpu.L4(),
            "A10G": gpu.A10G(),
            "A100": gpu.A100(),
        }
        
        # Update the function's GPU configuration
        train_job = app.function(
            image=train_bert4rec_job.image,
            gpu=gpu_map[gpu_type],
            volumes={"/data": train_bert4rec_job.volumes},
            timeout=60 * 60 * 6,
        )(train_bert4rec_job.f)
        
        # Run the training job
        result = train_job.remote(**config)
        
        # Store result
        self.results.append({
            "experiment_name": config["experiment_name"],
            "config": config,
            "result": result,
        })
        
        return result
    
    def run_all_experiments(self, experiments: List[Dict[str, Any]], gpu_type: str = "L4"):
        """
        Run multiple experiments sequentially.
        
        Args:
            experiments: List of experiment configurations
            gpu_type: GPU type to use for all experiments
        """
        print(f"Running {len(experiments)} experiments on Modal with {gpu_type} GPU")
        
        for exp_config in experiments:
            try:
                self.run_experiment(exp_config, gpu_type)
            except Exception as e:
                print(f"Error in experiment {exp_config['experiment_name']}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("All experiments completed!")
        print(f"{'='*60}")
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of all experiment results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\nExperiment Summary:")
        print("-" * 80)
        print(f"{'Experiment':<30} {'Test Recall@20':<15} {'Test NDCG@20':<15} {'Best Epoch':<10}")
        print("-" * 80)
        
        for exp in self.results:
            name = exp["experiment_name"][:30]
            metrics = exp["result"]["test_metrics"]
            recall = metrics.get("recall@20", 0)
            ndcg = metrics.get("ndcg@20", 0)
            best_epoch = exp["result"].get("best_epoch", 0)
            
            print(f"{name:<30} {recall:<15.4f} {ndcg:<15.4f} {best_epoch:<10}")
        
        print("-" * 80)
        
        # Find best performing experiment
        best_exp = max(self.results, key=lambda x: x["result"]["test_metrics"]["recall@20"])
        print(f"\nBest performing experiment: {best_exp['experiment_name']}")
        print(f"Test Recall@20: {best_exp['result']['test_metrics']['recall@20']:.4f}")
        print(f"Test NDCG@20: {best_exp['result']['test_metrics']['ndcg@20']:.4f}")
    
    def save_results(self, filepath: str):
        """Save all experiment results to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {filepath}")


def get_sample_experiments() -> List[Dict[str, Any]]:
    """
    Get a list of sample experiments with different hyperparameters.
    
    Returns:
        List of experiment configurations
    """
    runner = ExperimentRunner()
    experiments = []
    
    # Experiment 1: Baseline
    experiments.append(runner.create_experiment(
        name="baseline",
        n_epochs=10,
    ))
    
    # Experiment 2: Larger model
    experiments.append(runner.create_experiment(
        name="large_model",
        d_model=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=1024,
        n_epochs=10,
    ))
    
    # Experiment 3: Longer sequences
    experiments.append(runner.create_experiment(
        name="long_sequences",
        max_len=200,
        n_epochs=10,
    ))
    
    # Experiment 4: With prefix tokens
    experiments.append(runner.create_experiment(
        name="with_prefixes",
        add_channel_prefix=True,
        add_priceband_prefix=True,
        n_epochs=10,
    ))
    
    # Experiment 5: Different learning rate
    experiments.append(runner.create_experiment(
        name="lower_lr",
        learning_rate=5e-4,
        warmup_steps=2000,
        n_epochs=15,
    ))
    
    # Experiment 6: Higher masking probability
    experiments.append(runner.create_experiment(
        name="high_masking",
        mask_prob=0.25,
        n_epochs=10,
    ))
    
    # Experiment 7: Larger batch size
    experiments.append(runner.create_experiment(
        name="large_batch",
        batch_size=512,
        learning_rate=2e-3,
        n_epochs=10,
    ))
    
    return experiments


def run_hyperparameter_search():
    """
    Run a grid search over key hyperparameters.
    """
    runner = ExperimentRunner()
    experiments = []
    
    # Define hyperparameter grid
    learning_rates = [5e-4, 1e-3, 2e-3]
    model_sizes = [
        {"d_model": 128, "n_layers": 2, "dim_feedforward": 256},
        {"d_model": 256, "n_layers": 3, "dim_feedforward": 512},
        {"d_model": 512, "n_layers": 4, "dim_feedforward": 1024},
    ]
    mask_probs = [0.15, 0.20, 0.25]
    
    # Create experiments for grid search
    exp_id = 0
    for lr in learning_rates:
        for model_config in model_sizes:
            for mask_prob in mask_probs:
                exp_id += 1
                name = f"grid_{exp_id:03d}_lr{lr}_d{model_config['d_model']}_mask{mask_prob}"
                
                experiments.append(runner.create_experiment(
                    name=name,
                    learning_rate=lr,
                    mask_prob=mask_prob,
                    n_epochs=5,  # Shorter for grid search
                    **model_config
                ))
    
    print(f"Created {len(experiments)} experiments for grid search")
    return runner, experiments


def main():
    """Main entry point for running experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BERT4Rec experiments on Modal")
    parser.add_argument(
        "--mode",
        choices=["sample", "grid", "custom"],
        default="sample",
        help="Experiment mode: sample (predefined experiments), grid (hyperparameter search), or custom"
    )
    parser.add_argument(
        "--gpu",
        choices=["L4", "A10G", "A100"],
        default="L4",
        help="GPU type to use"
    )
    parser.add_argument(
        "--output",
        default="experiment_results.json",
        help="Output file for results"
    )
    
    # Custom experiment parameters
    parser.add_argument("--name", help="Custom experiment name")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--d-model", type=int, help="Model dimension")
    parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, help="Number of transformer layers")
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        # Run sample experiments
        runner = ExperimentRunner()
        experiments = get_sample_experiments()
        runner.run_all_experiments(experiments, gpu_type=args.gpu)
        runner.save_results(args.output)
        
    elif args.mode == "grid":
        # Run grid search
        runner, experiments = run_hyperparameter_search()
        runner.run_all_experiments(experiments, gpu_type=args.gpu)
        runner.save_results(args.output)
        
    elif args.mode == "custom":
        # Run custom experiment
        if not args.name:
            args.name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        runner = ExperimentRunner()
        config_overrides = {}
        
        if args.epochs:
            config_overrides["n_epochs"] = args.epochs
        if args.batch_size:
            config_overrides["batch_size"] = args.batch_size
        if args.lr:
            config_overrides["learning_rate"] = args.lr
        if args.d_model:
            config_overrides["d_model"] = args.d_model
        if args.n_heads:
            config_overrides["n_heads"] = args.n_heads
        if args.n_layers:
            config_overrides["n_layers"] = args.n_layers
        
        experiment = runner.create_experiment(args.name, **config_overrides)
        runner.run_experiment(experiment, gpu_type=args.gpu)
        runner.save_results(args.output)


if __name__ == "__main__":
    main()