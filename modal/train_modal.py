# modal/train_modal.py
"""
Modal training pipeline for BERT4Rec with 80:10:10 train/validation/test split.
This script loads data from Modal volumes and trains a BERT4Rec model.
"""

from modal import App, Image, Volume, Secret
import json
from pathlib import Path
from datetime import datetime
import os

app = App("bert4rec-train")

# Get the absolute path to the trainer directory
trainer_path = Path(__file__).parent.parent / "trainer"

# Build the Docker image with all required dependencies
image = (
    Image.debian_slim()
    .pip_install(
        "torch==2.3.1",
        "polars==1.5.0",
        "tqdm",
        "numpy",
    "wandb",
    )
    .add_local_dir(local_path=str(trainer_path), remote_path="/root/trainer")
)

# Mount the datasets volume
vol = Volume.from_name("datasets", create_if_missing=False)

# Load W&B secrets from local .env (WANDB_API_KEY, optional: WANDB_PROJECT, WANDB_ENTITY)
try:
    WANDB_SECRET = Secret.from_dotenv(".env", include=[
        "WANDB_API_KEY",
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ])
except Exception:
    WANDB_SECRET = None

def _run_bert4rec_impl(
    # Data parameters
    train_glob: str = "/data/modelling_data/*.parquet",
    
    # Model parameters
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 512,
    max_len: int = 100,
    dropout: float = 0.1,
    
    # Training parameters
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 10,
    warmup_steps: int = 1000,
    grad_clip_norm: float = 1.0,
    
    # Sequence parameters
    min_seq_len: int = 3,
    deduplicate_exact: bool = True,
    treat_same_day_as_basket: bool = True,
    
    # Prefix tokens (optional)
    add_segment_prefix: bool = False,
    add_channel_prefix: bool = False,
    add_priceband_prefix: bool = False,
    n_price_bins: int = 10,
    
    # Masking parameters
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    keep_original_prob: float = 0.10,
    
    # Split ratios
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    
    # Output
    out_dir: str = "/data/bert4rec_output",
    experiment_name: str = None,
    
    # Evaluation
    eval_topk: int = 20,
    eval_every_n_epochs: int = 2,
    
    # Random seed
    seed: int = 42,
):
    """
    Train BERT4Rec model with proper train/validation/test splits.
    
    The function performs:
    1. Data loading from parquet files
    2. Sequence preparation with optional prefix tokens
    3. 80:10:10 train/validation/test split
    4. Model training with MLM objective
    5. Periodic evaluation on validation and test sets
    6. Model and metrics saving
    """
    
    import torch
    import polars as pl
    from pathlib import Path
    from datetime import datetime
    import json
    from tqdm.auto import tqdm
    import sys
    import os

    # Weights & Biases monitoring - Optional
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        try:
            import wandb
            wandb_project = os.environ.get("WANDB_PROJECT", "bert4rec-modal")
            wandb_entity = os.environ.get("WANDB_ENTITY")
            wandb.init(
                project=wandb_project,
                entity=wandb_entity if wandb_entity else None,
                name=experiment_name,
                config={
                    "data": {
                        "train_glob": train_glob,
                        "train_ratio": train_ratio,
                        "valid_ratio": valid_ratio,
                        "test_ratio": test_ratio,
                    },
                    "model": {
                        "d_model": d_model,
                        "n_heads": n_heads,
                        "n_layers": n_layers,
                        "dim_feedforward": dim_feedforward,
                        "max_len": max_len,
                        "dropout": dropout,
                    },
                    "training": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "n_epochs": n_epochs,
                        "warmup_steps": warmup_steps,
                        "grad_clip_norm": grad_clip_norm,
                    },
                    "masking": {
                        "mask_prob": mask_prob,
                        "random_token_prob": random_token_prob,
                        "keep_original_prob": keep_original_prob,
                    },
                },
                tags=["modal", "bert4rec"],
            )
        except Exception as e:
            print(f"W&B init failed, continuing without logging: {e}")
            use_wandb = False
    
    # Import the BERT4Rec modules
    from trainer.bert4rec_modelling import (
        set_all_seeds,
        SequenceOptions,
        MaskingOptions,
        TrainConfig,
        TokenRegistry,
        prepare_sequences_with_polars,
        BERT4RecModel,
        BERT4RecDataset,
        NextItemEvalDataset,
        train_bert4rec,
        evaluate_mlm_loss,
        evaluate_next_item_topk,
    evaluate_next_item_at_ks,
    )
    
    # Set random seeds for reproducibility
    set_all_seeds(seed)
    
    # Setup experiment tracking
    if experiment_name is None:
        experiment_name = f"bert4rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_path = Path(out_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting BERT4Rec training experiment: {experiment_name}")
    print(f"Output directory: {output_path}")
    
    # ============================================
    # 1. Load Data
    # ============================================
    print("\n[1/6] Loading data from parquet files...")
    
    # Find all parquet files matching the glob pattern
    from glob import glob
    parquet_files = glob(train_glob)
    
    if not parquet_files:
        raise ValueError(f"No parquet files found matching pattern: {train_glob}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load each file based on its purpose
    transactions = None
    articles_features = None
    customer_segments = None
    
    print("Loading data files...")
    for file in parquet_files:
        file_path = Path(file)
        filename = file_path.name.lower()
        
        if 'transactions' in filename:
            transactions = pl.read_parquet(file)
            print(f"Loaded transactions: {len(transactions):,} rows, columns: {transactions.columns}")
        elif 'articles' in filename:
            articles_features = pl.read_parquet(file)
            print(f"Loaded articles features: {len(articles_features):,} rows, columns: {articles_features.columns}")
        elif 'customers' in filename or 'segmented' in filename:
            customer_segments = pl.read_parquet(file)
            print(f"Loaded customer segments: {len(customer_segments):,} rows, columns: {customer_segments.columns}")
        else:
            print(f"Unknown file type: {filename}, skipping...")
    
    if transactions is None:
        raise ValueError("No transactions file found! Expected a file with 'transactions' in the name.")
    
    print(f"Loaded {len(transactions):,} transactions")
    print(f"Unique customers: {transactions['customer_id'].n_unique():,}")
    print(f"Unique articles: {transactions['article_id'].n_unique():,}")
    
    # Optional: Load user segments if available and requested
    user_segments = None
    if add_segment_prefix:
        segment_path = Path("/data/user_segments.parquet")
        if segment_path.exists():
            user_segments = pl.read_parquet(segment_path)
            print(f"Loaded user segments for {len(user_segments):,} customers")
        else:
            print("Warning: add_segment_prefix=True but no user_segments.parquet found")
            add_segment_prefix = False
    
    # ============================================
    # 2. Prepare Sequences
    # ============================================
    print("\n[2/6] Preparing sequences...")
    
    sequence_options = SequenceOptions(
        max_len=max_len,
        min_len=min_seq_len,
        deduplicate_exact=deduplicate_exact,
        treat_same_day_as_basket=treat_same_day_as_basket,
        add_segment_prefix=add_segment_prefix,
        add_channel_prefix=add_channel_prefix,
        add_priceband_prefix=add_priceband_prefix,
        n_price_bins=n_price_bins,
    )
    
    prepared = prepare_sequences_with_polars(
        transactions=transactions,
        user_segments=user_segments,
        options=sequence_options,
    )
    
    print(f"Prepared {len(prepared.sequences):,} sequences")
    print(f"Vocabulary size: {prepared.registry.vocab_size:,}")
    print(f"Average sequence length: {sum(len(s) for s in prepared.sequences) / len(prepared.sequences):.1f}")
    
    # ============================================
    # 3. Create Train/Validation/Test Splits
    # ============================================
    print(f"\n[3/6] Creating {int(train_ratio*100)}:{int(valid_ratio*100)}:{int(test_ratio*100)} train/val/test split...")
    
    import random
    
    n_sequences = len(prepared.sequences)
    indices = list(range(n_sequences))
    random.shuffle(indices)
    
    # Calculate split points
    n_train = int(n_sequences * train_ratio)
    n_valid = int(n_sequences * valid_ratio)
    n_test = n_sequences - n_train - n_valid
    
    # Split indices
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]
    
    print(f"Train: {len(train_indices):,} sequences")
    print(f"Valid: {len(valid_indices):,} sequences")
    print(f"Test:  {len(test_indices):,} sequences")
    
    # Create datasets
    def get_sequences_by_indices(indices):
        seqs = [prepared.sequences[i] for i in indices]
        prefix_lens = [prepared.prefix_lengths[i] for i in indices]
        return seqs, prefix_lens
    
    train_seqs, train_prefix_lens = get_sequences_by_indices(train_indices)
    valid_seqs, valid_prefix_lens = get_sequences_by_indices(valid_indices)
    test_seqs, test_prefix_lens = get_sequences_by_indices(test_indices)
    
    # ============================================
    # 4. Create PyTorch Datasets and DataLoaders
    # ============================================
    print("\n[4/6] Creating datasets and dataloaders...")
    
    masking_options = MaskingOptions(
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        keep_original_prob=keep_original_prob,
    )
    
    # Training dataset (with masking)
    train_dataset = BERT4RecDataset(
        sequences=train_seqs,
        prefix_lengths=train_prefix_lens,
        vocab_size=prepared.registry.vocab_size,
        max_len=max_len,
        masking=masking_options,
    )
    
    # Validation dataset (with masking for MLM loss)
    valid_dataset = BERT4RecDataset(
        sequences=valid_seqs,
        prefix_lengths=valid_prefix_lens,
        vocab_size=prepared.registry.vocab_size,
        max_len=max_len,
        masking=masking_options,
    )
    
    # Test datasets
    # MLM evaluation dataset
    test_mlm_dataset = BERT4RecDataset(
        sequences=test_seqs,
        prefix_lengths=test_prefix_lens,
        vocab_size=prepared.registry.vocab_size,
        max_len=max_len,
        masking=masking_options,
    )
    
    # Next-item prediction evaluation dataset
    test_next_item_dataset = NextItemEvalDataset(
        sequences=test_seqs,
        prefix_lengths=test_prefix_lens,
        max_len=max_len,
    )
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    test_mlm_loader = DataLoader(
        test_mlm_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    test_next_item_loader = DataLoader(
        test_next_item_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # ============================================
    # 5. Initialise Model
    # ============================================
    print("\n[5/6] Initialising BERT4Rec model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BERT4RecModel(
        vocab_size=prepared.registry.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============================================
    # 6. Training Loop
    # ============================================
    print(f"\n[6/6] Starting training for {n_epochs} epochs...")
    
    # Setup optimiser and scheduler
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Calculate total training steps
    total_steps = n_epochs * len(train_loader)
    
    # Learning rate scheduler
    from trainer.bert4rec_modelling import WarmupLinearSchedule
    scheduler = WarmupLinearSchedule(
        optimiser,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training history
    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_recall": [],
        "valid_ndcg": [],
        "test_loss": [],
        "test_recall": [],
        "test_ndcg": [],
    }
    
    best_valid_recall = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Epoch {epoch}/{n_epochs} ---")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            total=len(train_loader),
            dynamic_ncols=True,
            ncols=100,
            mininterval=0.5,
            smoothing=0.1,
            leave=False,
            ascii=True,
            file=sys.stdout,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimiser.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            B, L, V = logits.size()
            loss = criterion(logits.view(B * L, V), labels.view(B * L))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Optimiser step
            optimiser.step()
            scheduler.step()
            
            # Track loss
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_lr()[0]:.6f}'
            })
        
        avg_train_loss = train_loss / train_steps
        history["train_loss"].append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        if use_wandb:
            try:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "lr": scheduler.get_lr()[0] if hasattr(scheduler, "get_lr") else optimiser.param_groups[0]["lr"],
                }, step=epoch)
            except Exception:
                pass
        
        # Validation phase
        if epoch % eval_every_n_epochs == 0 or epoch == n_epochs:
            print("\nEvaluating on validation set...")
            
            # MLM loss on validation
            valid_loss = evaluate_mlm_loss(model, valid_loader, device)
            history["valid_loss"].append(valid_loss)
            print(f"Validation MLM loss: {valid_loss:.4f}")
            
            # Next-item prediction on validation
            valid_next_item_loader = DataLoader(
                NextItemEvalDataset(valid_seqs, valid_prefix_lens, max_len),
                batch_size=batch_size,
                shuffle=False,
            )
            
            valid_recall, valid_ndcg = evaluate_next_item_topk(
                model, valid_next_item_loader, device,
                prepared.registry, topk=eval_topk
            )
            history["valid_recall"].append(valid_recall)
            history["valid_ndcg"].append(valid_ndcg)
            print(f"Validation Recall@{eval_topk}: {valid_recall:.4f}")
            print(f"Validation NDCG@{eval_topk}: {valid_ndcg:.4f}")
            if use_wandb:
                try:
                    wandb.log({
                        f"valid/recall@{eval_topk}": valid_recall,
                        f"valid/ndcg@{eval_topk}": valid_ndcg,
                        "valid/loss": valid_loss,
                    }, step=epoch)
                except Exception:
                    pass
            
            # Save best model
            if valid_recall > best_valid_recall:
                best_valid_recall = valid_recall
                best_epoch = epoch
                
                # Save model checkpoint
                checkpoint_path = output_path / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'valid_recall': valid_recall,
                    'valid_ndcg': valid_ndcg,
                    'config': {
                        'd_model': d_model,
                        'n_heads': n_heads,
                        'n_layers': n_layers,
                        'dim_feedforward': dim_feedforward,
                        'max_len': max_len,
                        'dropout': dropout,
                        'vocab_size': prepared.registry.vocab_size,
                    }
                }, checkpoint_path)
                print(f"Saved best model checkpoint (epoch {epoch})")
                if use_wandb:
                    try:
                        wandb.run.summary["best/epoch"] = epoch
                        wandb.run.summary["best/valid_recall@%d" % eval_topk] = valid_recall
                        wandb.run.summary["best/valid_ndcg@%d" % eval_topk] = valid_ndcg
                    except Exception:
                        pass
    
    print(f"\nBest validation Recall@{eval_topk}: {best_valid_recall:.4f} at epoch {best_epoch}")
    
    # ============================================
    # 7. Final Test Set Evaluation
    # ============================================
    print("\n=== Final Test Set Evaluation ===")
    
    # Load best model
    checkpoint = torch.load(output_path / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test MLM loss
    test_loss = evaluate_mlm_loss(model, test_mlm_loader, device)
    print(f"Test MLM loss: {test_loss:.4f}")
    
    # Test next-item prediction
    test_recall, test_ndcg = evaluate_next_item_topk(
        model, test_next_item_loader, device,
        prepared.registry, topk=eval_topk
    )
    print(f"Test Recall@{eval_topk}: {test_recall:.4f}")
    print(f"Test NDCG@{eval_topk}: {test_ndcg:.4f}")
    
    # Multi-K metrics
    multi_k = evaluate_next_item_at_ks(
        model, test_next_item_loader, device, ks=(5, 10, 20, 50)
    )
    print("\nMulti-K metrics:")
    for k in (5, 10, 20, 50):
        print(
            f"k={k} | Acc@1={multi_k.get('accuracy@1', 0):.4f} "
            f"Prec@{k}={multi_k.get(f'precision@{k}', 0):.4f} "
            f"Rec@{k}={multi_k.get(f'recall@{k}', 0):.4f} "
            f"F1@{k}={multi_k.get(f'f1@{k}', 0):.4f} "
            f"NDCG@{k}={multi_k.get(f'ndcg@{k}', 0):.4f}"
        )
    if use_wandb:
        try:
            payload = {
                "test/mlm_loss": test_loss,
                f"test/recall@{eval_topk}": test_recall,
                f"test/ndcg@{eval_topk}": test_ndcg,
                "test/accuracy@1": multi_k.get("accuracy@1"),
            }
            for k in (5, 10, 20, 50):
                payload[f"test/precision@{k}"] = multi_k.get(f"precision@{k}")
                payload[f"test/recall@{k}"] = multi_k.get(f"recall@{k}")
                payload[f"test/f1@{k}"] = multi_k.get(f"f1@{k}")
                payload[f"test/ndcg@{k}"] = multi_k.get(f"ndcg@{k}")
            wandb.log(payload)
            wandb.run.summary[f"test/recall@{eval_topk}"] = test_recall
            wandb.run.summary[f"test/ndcg@{eval_topk}"] = test_ndcg
            wandb.run.summary["test/accuracy@1"] = multi_k.get("accuracy@1")
            for k in (5, 10, 20, 50):
                wandb.run.summary[f"test/precision@{k}"] = multi_k.get(f"precision@{k}")
                wandb.run.summary[f"test/recall@{k}"] = multi_k.get(f"recall@{k}")
                wandb.run.summary[f"test/f1@{k}"] = multi_k.get(f"f1@{k}")
                wandb.run.summary[f"test/ndcg@{k}"] = multi_k.get(f"ndcg@{k}")
        except Exception:
            pass
    
    # Save final results
    results = {
        "experiment_name": experiment_name,
        "best_epoch": best_epoch,
        "test_metrics": {
            "mlm_loss": test_loss,
            f"recall@{eval_topk}": test_recall,
            f"ndcg@{eval_topk}": test_ndcg,
            **multi_k,
        },
        "best_valid_metrics": {
            f"recall@{eval_topk}": best_valid_recall,
            f"ndcg@{eval_topk}": history["valid_ndcg"][-1] if history["valid_ndcg"] else 0,
        },
        "config": {
            "model": {
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "dim_feedforward": dim_feedforward,
                "max_len": max_len,
                "dropout": dropout,
                "vocab_size": prepared.registry.vocab_size,
            },
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "n_epochs": n_epochs,
                "warmup_steps": warmup_steps,
                "grad_clip_norm": grad_clip_norm,
            },
            "data": {
                "train_size": len(train_indices),
                "valid_size": len(valid_indices),
                "test_size": len(test_indices),
                "min_seq_len": min_seq_len,
                "mask_prob": mask_prob,
            },
        },
        "history": history,
    }
    
    # Save results to JSON
    results_path = output_path / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Save the token registry for inference
    registry_path = output_path / "token_registry.json"
    with open(registry_path, 'w') as f:
        json.dump({
            "item2id": {str(k): v for k, v in prepared.registry.item2id.items()},
            "id2item": {str(k): v for k, v in prepared.registry.id2item.items()},
            "prefix_token2id": prepared.registry.prefix_token2id,
            "vocab_size": prepared.registry.vocab_size,
        }, f, indent=2)
    
    print(f"Token registry saved to: {registry_path}")
    
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass
    return results


# GPU-specific Modal functions wrapping the shared implementation
@app.function(
    image=image,
    gpu="A100",  # A100 40GB
    volumes={"/data": vol},
    timeout=60 * 60 * 6,
    secrets=[WANDB_SECRET] if WANDB_SECRET else [],
)
def train_bert4rec_job(
    train_glob: str = "/data/modelling_data/*.parquet",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 512,
    max_len: int = 100,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 10,
    warmup_steps: int = 1000,
    grad_clip_norm: float = 1.0,
    min_seq_len: int = 3,
    deduplicate_exact: bool = True,
    treat_same_day_as_basket: bool = True,
    add_segment_prefix: bool = False,
    add_channel_prefix: bool = False,
    add_priceband_prefix: bool = False,
    n_price_bins: int = 10,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    keep_original_prob: float = 0.10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    out_dir: str = "/data/bert4rec_output",
    experiment_name: str = None,
    eval_topk: int = 20,
    eval_every_n_epochs: int = 2,
    seed: int = 42,
):
    return _run_bert4rec_impl(
        train_glob=train_glob,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
        grad_clip_norm=grad_clip_norm,
        min_seq_len=min_seq_len,
        deduplicate_exact=deduplicate_exact,
        treat_same_day_as_basket=treat_same_day_as_basket,
        add_segment_prefix=add_segment_prefix,
        add_channel_prefix=add_channel_prefix,
        add_priceband_prefix=add_priceband_prefix,
        n_price_bins=n_price_bins,
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        keep_original_prob=keep_original_prob,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        out_dir=out_dir,
        experiment_name=experiment_name,
        eval_topk=eval_topk,
        eval_every_n_epochs=eval_every_n_epochs,
        seed=seed,
    )


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    timeout=60 * 60 * 6,
    secrets=[WANDB_SECRET] if WANDB_SECRET else [],
)
def train_bert4rec_job_a10g(
    train_glob: str = "/data/modelling_data/*.parquet",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 512,
    max_len: int = 100,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 10,
    warmup_steps: int = 1000,
    grad_clip_norm: float = 1.0,
    min_seq_len: int = 3,
    deduplicate_exact: bool = True,
    treat_same_day_as_basket: bool = True,
    add_segment_prefix: bool = False,
    add_channel_prefix: bool = False,
    add_priceband_prefix: bool = False,
    n_price_bins: int = 10,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    keep_original_prob: float = 0.10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    out_dir: str = "/data/bert4rec_output",
    experiment_name: str = None,
    eval_topk: int = 20,
    eval_every_n_epochs: int = 2,
    seed: int = 42,
):
    return _run_bert4rec_impl(
        train_glob=train_glob,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
        grad_clip_norm=grad_clip_norm,
        min_seq_len=min_seq_len,
        deduplicate_exact=deduplicate_exact,
        treat_same_day_as_basket=treat_same_day_as_basket,
        add_segment_prefix=add_segment_prefix,
        add_channel_prefix=add_channel_prefix,
        add_priceband_prefix=add_priceband_prefix,
        n_price_bins=n_price_bins,
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        keep_original_prob=keep_original_prob,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        out_dir=out_dir,
        experiment_name=experiment_name,
        eval_topk=eval_topk,
        eval_every_n_epochs=eval_every_n_epochs,
        seed=seed,
    )


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol},
    timeout=60 * 60 * 6,
    secrets=[WANDB_SECRET] if WANDB_SECRET else [],
)
def train_bert4rec_job_l4(
    train_glob: str = "/data/modelling_data/*.parquet",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 512,
    max_len: int = 100,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 10,
    warmup_steps: int = 1000,
    grad_clip_norm: float = 1.0,
    min_seq_len: int = 3,
    deduplicate_exact: bool = True,
    treat_same_day_as_basket: bool = True,
    add_segment_prefix: bool = False,
    add_channel_prefix: bool = False,
    add_priceband_prefix: bool = False,
    n_price_bins: int = 10,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    keep_original_prob: float = 0.10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    out_dir: str = "/data/bert4rec_output",
    experiment_name: str = None,
    eval_topk: int = 20,
    eval_every_n_epochs: int = 2,
    seed: int = 42,
):
    return _run_bert4rec_impl(
        train_glob=train_glob,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
        grad_clip_norm=grad_clip_norm,
        min_seq_len=min_seq_len,
        deduplicate_exact=deduplicate_exact,
        treat_same_day_as_basket=treat_same_day_as_basket,
        add_segment_prefix=add_segment_prefix,
        add_channel_prefix=add_channel_prefix,
        add_priceband_prefix=add_priceband_prefix,
        n_price_bins=n_price_bins,
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        keep_original_prob=keep_original_prob,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        out_dir=out_dir,
        experiment_name=experiment_name,
        eval_topk=eval_topk,
        eval_every_n_epochs=eval_every_n_epochs,
        seed=seed,
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/data": vol},
    timeout=60 * 60 * 6,
    secrets=[WANDB_SECRET] if WANDB_SECRET else [],
)
def train_bert4rec_job_a100_80gb(
    train_glob: str = "/data/modelling_data/*.parquet",
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 512,
    max_len: int = 100,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    n_epochs: int = 10,
    warmup_steps: int = 1000,
    grad_clip_norm: float = 1.0,
    min_seq_len: int = 3,
    deduplicate_exact: bool = True,
    treat_same_day_as_basket: bool = True,
    add_segment_prefix: bool = False,
    add_channel_prefix: bool = False,
    add_priceband_prefix: bool = False,
    n_price_bins: int = 10,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.10,
    keep_original_prob: float = 0.10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    out_dir: str = "/data/bert4rec_output",
    experiment_name: str = None,
    eval_topk: int = 20,
    eval_every_n_epochs: int = 2,
    seed: int = 42,
):
    return _run_bert4rec_impl(
        train_glob=train_glob,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs,
        warmup_steps=warmup_steps,
        grad_clip_norm=grad_clip_norm,
        min_seq_len=min_seq_len,
        deduplicate_exact=deduplicate_exact,
        treat_same_day_as_basket=treat_same_day_as_basket,
        add_segment_prefix=add_segment_prefix,
        add_channel_prefix=add_channel_prefix,
        add_priceband_prefix=add_priceband_prefix,
        n_price_bins=n_price_bins,
        mask_prob=mask_prob,
        random_token_prob=random_token_prob,
        keep_original_prob=keep_original_prob,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        out_dir=out_dir,
        experiment_name=experiment_name,
        eval_topk=eval_topk,
        eval_every_n_epochs=eval_every_n_epochs,
        seed=seed,
    )


@app.local_entrypoint()
def main(
    # Add command line arguments
    experiment_name: str = None,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    gpu_type: str = "L4",  # L4, A10G, A100 (40GB), or A100-80GB
):
    """
    Local entrypoint for running the training job on Modal.
    
    Usage:
        modal run modal/train_modal.py --epochs 20 --batch-size 512
    """
    
    print(f"Launching BERT4Rec training on Modal with GPU type: {gpu_type}")
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    
    # Dispatch to the Modal function that matches the requested GPU type
    def _normalise(resource: str):
        return (resource or "").strip().lower().replace(" ", "")

    r = _normalise(gpu_type)
    if r in {"a100-80gb", "a100-80g", "a10080", "a10080gb"}:
        fn = train_bert4rec_job_a100_80gb
        resolved = "A100-80GB"
    elif r in {"a10g"}:
        fn = train_bert4rec_job_a10g
        resolved = "A10G"
    elif r in {"l4"}:
        fn = train_bert4rec_job_l4
        resolved = "L4"
    else:
        fn = train_bert4rec_job
        resolved = "A100"

    print(f"Resolved GPU resource: {resolved}")

    # Run the training job on the selected GPU
    results = fn.remote(
        experiment_name=experiment_name,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    print("\nTraining completed!")
    print(f"Test Recall@20: {results['test_metrics']['recall@20']:.4f}")
    print(f"Test NDCG@20: {results['test_metrics']['ndcg@20']:.4f}")
    
    return results


if __name__ == "__main__":
    # This allows running with: python modal/train_modal.py
    import sys
    from modal import runner
    runner.main(sys.argv)