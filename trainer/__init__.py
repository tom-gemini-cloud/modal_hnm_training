"""
Trainer module for H&M data analytics project.

This module contains training implementations for various machine learning models
used in the H&M customer data analysis project.
"""

from .bert4rec_modelling import (
    # Core data structures
    TokenRegistry,
    SequenceOptions,
    PreparedData,
    MaskingOptions,
    TrainConfig,
    
    # Model and datasets
    BERT4RecModel,
    BERT4RecDataset,
    NextItemEvalDataset,
    
    # Main functions
    set_all_seeds,
    prepare_sequences_with_polars,
    train_bert4rec,
    train,  # Main end-to-end training function
    evaluate_mlm_loss,
    evaluate_next_item_topk,
    build_dataloaders_for_bert4rec,
    
    # Learning rate scheduler
    WarmupLinearSchedule,
)

__all__ = [
    "TokenRegistry",
    "SequenceOptions", 
    "PreparedData",
    "MaskingOptions",
    "TrainConfig",
    "BERT4RecModel",
    "BERT4RecDataset",
    "NextItemEvalDataset",
    "set_all_seeds",
    "prepare_sequences_with_polars",
    "train_bert4rec",
    "train",  # Main end-to-end training function
    "evaluate_mlm_loss",
    "evaluate_next_item_topk", 
    "build_dataloaders_for_bert4rec",
    "WarmupLinearSchedule",
]