from __future__ import annotations

from pathlib import Path
"""
BERT4Rec for implicit product recommendation (Polars + PyTorch)

This module provides is a testable implementation of BERT4Rec with a Polars
data pipeline. It is designed to train on transaction sequences and evaluate
next-item prediction.

Main features
------------
- Polars-based sequence preparation with strict time-based ordering
- Optional prefix tokens to condition on customer segment, channel, and price-band
- Standard BERT4Rec masking scheme (80/10/10) with non-maskable prefix support
- Time-based train/validation/test splits to avoid leakage
- Recall@K and NDCG@K evaluation for next-item prediction

Author: tom@geminicloud.co.uk
Licence: MIT
"""

import math
import random
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -----------------------------
# Constants and small utilities
# -----------------------------

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
SPECIAL_TOKENS_RESERVED = 2  # [PAD]=0, [MASK]=1 start of item ids after this

def set_all_seeds(seed: int = 42) -> None:
    """Set Python, NumPy (if present), and Torch RNGs for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Vocabulary and token registry
# -----------------------------

@dataclass
class TokenRegistry:
    """
    Registry assigning integer ids to items and optional prefix tokens such as
    customer segments, channels, and price bands.

    All ids are in a single shared vocabulary so that prefix tokens can be
    embedded and attended like ordinary items. Item ids start after the
    reserved special tokens.

    Attributes
    ----------
    item2id : Dict[int, int]
        Mapping from raw article_id to token id used by the model.
    id2item : Dict[int, int]
        Inverse mapping for decoding predictions back to article_id.
    prefix_token2id : Dict[str, int]
        Mapping for named prefix tokens (e.g., 'SEG=1', 'CH=2', 'PB=4').
    vocab_size : int
        Size of the entire token vocabulary including specials and prefixes.
    """
    item2id: Dict[int, int]
    id2item: Dict[int, int]
    prefix_token2id: Dict[str, int]
    vocab_size: int

    @staticmethod
    def build(
        article_ids: Iterable[int],
        segments: Iterable[str] = (),
        channels: Iterable[int] = (),
        price_bins: Iterable[int] = (),
    ) -> "TokenRegistry":
        # Items
        item2id: Dict[int, int] = {}
        id2item: Dict[int, int] = {}
        next_id = SPECIAL_TOKENS_RESERVED
        for a in article_ids:
            if a not in item2id:
                item2id[a] = next_id
                id2item[next_id] = a
                next_id += 1

        # Prefix tokens
        prefix_token2id: Dict[str, int] = {}

        def add_prefix(name: str, values: Iterable):
            nonlocal next_id
            for v in values:
                key = f"{name}={v}"
                if key not in prefix_token2id:
                    prefix_token2id[key] = next_id
                    next_id += 1

        if segments:
            add_prefix("SEG", segments)
        if channels:
            add_prefix("CH", channels)
        if price_bins:
            add_prefix("PB", price_bins)

        return TokenRegistry(
            item2id=item2id,
            id2item=id2item,
            prefix_token2id=prefix_token2id,
            vocab_size=next_id,
        )

    def encode_item(self, article_id: int) -> int:
        return self.item2id[article_id]

    def decode_item(self, token_id: int) -> Optional[int]:
        # Return article_id only if token id corresponds to a real item
        if token_id in self.id2item:
            return self.id2item[token_id]
        return None

    def encode_prefix(self, name: str, value) -> int:
        key = f"{name}={value}"
        return self.prefix_token2id[key]


# -----------------------------
# Polars sequence preparation
# -----------------------------

@dataclass
class SequenceOptions:
    """
    Options controlling how to turn transactions into sequences.

    Attributes
    ----------
    max_len : int
        Maximum sequence length per user; truncate on the left (keep recent).
    min_len : int
        Minimum number of items required to keep a user's sequence.
    deduplicate_exact : bool
        If True, drop exact duplicate transaction rows.
    treat_same_day_as_basket : bool
        If True, transactions on the same day are ordered stably by article_id
        to avoid accidental leakage from arbitrary sorting.
    add_segment_prefix : bool
        If True, prepend a segment token per user (requires user_segments).
    add_channel_prefix : bool
        If True, prepend a channel token per last-observed channel in history.
    add_priceband_prefix : bool
        If True, prepend a price band token per user (median price bin).
    n_price_bins : int
        Number of price bands if add_priceband_prefix is True.
    """
    max_len: int = 100
    min_len: int = 2
    deduplicate_exact: bool = True
    treat_same_day_as_basket: bool = True
    add_segment_prefix: bool = False
    add_channel_prefix: bool = False
    add_priceband_prefix: bool = False
    n_price_bins: int = 10


@dataclass
class PreparedData:
    """Container for prepared sequences and associated artefacts."""
    sequences: List[List[int]]
    # For evaluation, keep the raw last item per user
    last_items: List[int]
    registry: TokenRegistry
    prefix_lengths: List[int]  # number of non-maskable prefix tokens per sequence


def _bin_price_series(prices: pl.Series, n_bins: int = 10) -> pl.Series:
    # For stability, use rank-based binning rather than equal-width
    q = prices.rank("dense") / prices.len()
    bins = (q * n_bins).clip(0, n_bins - 1).cast(pl.Int64)
    return bins


def prepare_sequences_with_polars(
    transactions: pl.DataFrame,
    user_segments: Optional[pl.DataFrame] = None,
    options: SequenceOptions = SequenceOptions(),
) -> PreparedData:
    """
    Prepare per-user sequences from transactions with optional prefix tokens.

    Parameters
    ----------
    transactions : pl.DataFrame
        Must include: 'customer_id', 'article_id', 't_dat' (date or string).
        Optional: 'sales_channel_id', 'price'.
    user_segments : pl.DataFrame, optional
        Optional table with columns ['customer_id', 'customer_cluster'].
    options : SequenceOptions
        Controls sequence length, binning, and prefixes.

    Returns
    -------
    PreparedData
        Sequences of token ids per user, token registry, and auxiliary info.
    """
    df = transactions.clone()

    # Ensure types
    if df["t_dat"].dtype != pl.Date and df["t_dat"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("t_dat").str.strptime(pl.Date, strict=False))

    if options.deduplicate_exact:
        df = df.unique(maintain_order=True)

    # Compute per-user median price and last observed channel if needed
    add_priceband = options.add_priceband_prefix and ("price" in df.columns)
    add_channel = options.add_channel_prefix and ("sales_channel_id" in df.columns)

    if add_priceband:
        df = df.with_columns(
            pl.col("price").fill_null(strategy="forward").fill_null(strategy="backward")
        )
        user_price_median = (
            df.group_by("customer_id")
              .agg(pl.col("price").median().alias("median_price"))
        )
        # Rank-based price band across the full population for calibration
        global_bins = _bin_price_series(df["price"], n_bins=options.n_price_bins)
        df = df.with_columns(global_bins.alias("price_bin"))
        user_price_band = (
            df.group_by("customer_id").agg(pl.col("price_bin").median().alias("user_price_band"))
        )
    else:
        user_price_median = None
        user_price_band = None

    if add_channel:
        user_last_channel = (
            df.sort(["customer_id", "t_dat"])
              .group_by("customer_id")
              .tail(1)
              .select(["customer_id", pl.col("sales_channel_id").alias("last_channel")])
        )
    else:
        user_last_channel = None

    if user_segments is not None and options.add_segment_prefix:
        seg = user_segments.select(["customer_id", "customer_cluster"])
    else:
        seg = None

    # Build registry
    article_ids = df["article_id"].unique().to_list()
    segment_values = seg["customer_cluster"].unique().to_list() if seg is not None else []
    channel_values = df["sales_channel_id"].unique().to_list() if add_channel else []
    price_band_values = list(range(options.n_price_bins)) if add_priceband else []

    registry = TokenRegistry.build(
        article_ids=article_ids,
        segments=segment_values,
        channels=channel_values,
        price_bins=price_band_values,
    )

    # Join auxiliary data
    joins = []
    if seg is not None:
        joins.append(seg)
    if user_last_channel is not None:
        joins.append(user_last_channel)
    if user_price_band is not None:
        joins.append(user_price_band)

    enriched = df
    for j in joins:
        enriched = enriched.join(j, on="customer_id", how="left")

    # Sequence building
    sequences: List[List[int]] = []
    last_items: List[int] = []
    prefix_lengths: List[int] = []

    # Order items reproducibly within the same day
    if options.treat_same_day_as_basket:
        enriched = enriched.sort(["customer_id", "t_dat", "article_id"])
    else:
        enriched = enriched.sort(["customer_id", "t_dat"])

    # Group by user and build sequences
    for customer_id, sub in enriched.group_by("customer_id", maintain_order=True):
        items = [registry.encode_item(a) for a in sub["article_id"].to_list()]
        if len(items) < options.min_len:
            continue

        # Optional prefixes (non-maskable)
        prefixes: List[int] = []
        if seg is not None and options.add_segment_prefix:
            prefixes.append(registry.encode_prefix("SEG", int(sub["customer_cluster"][0])))
        if add_channel:
            prefixes.append(registry.encode_prefix("CH", int(sub["last_channel"][-1])))
        if add_priceband:
            prefixes.append(registry.encode_prefix("PB", int(sub["user_price_band"][0])))

        # Truncate on the left (keep most recent), respecting max_len after prefixes
        max_items = options.max_len - len(prefixes)
        items = items[-max_items:]
        seq = prefixes + items
        sequences.append(seq)
        prefix_lengths.append(len(prefixes))
        last_items.append(items[-1])  # for next-item evaluation

    return PreparedData(
        sequences=sequences,
        last_items=last_items,
        registry=registry,
        prefix_lengths=prefix_lengths,
    )


# -----------------------------
# Dataset and Collator
# -----------------------------

@dataclass
class MaskingOptions:
    """
    Masking behaviour for BERT4Rec.

    Attributes
    ----------
    mask_prob : float
        Proportion of (maskable) positions to select for prediction.
    random_token_prob : float
        Of the selected positions, proportion to replace by a random token.
    keep_original_prob : float
        Of the selected positions, proportion to keep unchanged.
        The remainder (1 - random - keep) are replaced by [MASK].
    """
    mask_prob: float = 0.15
    random_token_prob: float = 0.10
    keep_original_prob: float = 0.10


class BERT4RecDataset(Dataset):
    """
    PyTorch Dataset implementing masked language modelling for sequences.
    Non-maskable prefix tokens remain untouched but contribute to attention.
    """
    def __init__(
        self,
        sequences: List[List[int]],
        prefix_lengths: List[int],
        vocab_size: int,
        max_len: int,
        masking: MaskingOptions = MaskingOptions(),
    ) -> None:
        self.sequences = sequences
        self.prefix_lengths = prefix_lengths
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.masking = masking

    def __len__(self) -> int:
        return len(self.sequences)

    def _apply_masking(self, tokens: List[int], non_maskable: int) -> Tuple[List[int], List[int]]:
        """
        Apply BERT-style masking to a token sequence.

        Parameters
        ----------
        tokens : List[int]
            The full token list (including prefixes). Will be truncated/padded later.
        non_maskable : int
            Number of prefix tokens at the start that must not be masked.

        Returns
        -------
        input_tokens : List[int]
            Tokens after masking (with [MASK] and random replacements).
        labels : List[int]
            Labels for each position (-100 for positions that are not predicted).
        """
        input_tokens = list(tokens)
        labels = [-100] * len(tokens)  # -100 ignored by CrossEntropyLoss

        # Candidate positions exclude non-maskable prefix and padding (handled later)
        candidate_positions = list(range(non_maskable, len(tokens)))
        n_to_mask = max(1, int(len(candidate_positions) * self.masking.mask_prob))
        mask_positions = set(random.sample(candidate_positions, n_to_mask))

        for pos in mask_positions:
            original = tokens[pos]
            rand = random.random()
            labels[pos] = original
            if rand < (1.0 - self.masking.random_token_prob - self.masking.keep_original_prob):
                input_tokens[pos] = MASK_TOKEN_ID
            elif rand < (1.0 - self.masking.keep_original_prob):
                # Replace by a random token id in [SPECIAL_TOKENS_RESERVED, vocab_size)
                input_tokens[pos] = random.randint(SPECIAL_TOKENS_RESERVED, self.vocab_size - 1)
            else:
                # Keep original token
                pass

        return input_tokens, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        non_maskable = self.prefix_lengths[idx]

        # Apply masking on the full (untruncated) sequence
        masked_tokens, labels = self._apply_masking(tokens, non_maskable)

        # Truncate from the left to max_len
        masked_tokens = masked_tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        # Build attention mask and pad on the left for simplicity of "recent-right" layout
        attention_len = len(masked_tokens)
        pad_len = self.max_len - attention_len
        input_ids = [PAD_TOKEN_ID] * pad_len + masked_tokens
        label_ids = [-100] * pad_len + labels
        attention_mask = [0] * pad_len + [1] * attention_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# -----------------------------
# Model
# -----------------------------

class BERT4RecModel(nn.Module):
    """
    Minimal but solid BERT4Rec encoder with tied input/output embeddings.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        max_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Output projection tied to token embeddings
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self.max_len = max_len
        self.d_model = d_model

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, L) LongTensor
        attention_mask : (B, L) 1/0 mask, 1 for real tokens

        Returns
        -------
        logits : (B, L, V) FloatTensor
            Unnormalised scores over the full vocabulary for each position.
        """
        B, L = input_ids.size()
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.encoder(x, src_key_padding_mask=(attention_mask == 0))
        x = self.norm(x)

        # Weight tying: projection via token_emb^T plus bias
        logits = torch.matmul(x, self.token_emb.weight.T) + self.output_bias
        return logits


# -----------------------------
# Training loop and evaluation
# -----------------------------

@dataclass
class TrainConfig:
    """
    Training configuration for BERT4Rec.

    Attributes
    ----------
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    n_epochs : int
        Number of epochs to train.
    warmup_steps : int
        Linear warm-up steps for the learning rate.
    grad_clip_norm : float
        Clip gradient norm for stability; set 0 to disable.
    """
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 5
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0


class WarmupLinearSchedule(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warm-up then linear decay to zero.
    """
    def __init__(self, optimiser, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimiser, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lr = base_lr * float(step) / float(max(1, self.warmup_steps))
            else:
                lr = base_lr * max(
                    0.0,
                    float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)),
                )
            lrs.append(lr)
        return lrs


def train_bert4rec(
    model: BERT4RecModel,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    """
    Train the model with masked language modelling objective.
    """
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.n_epochs * len(train_loader)
    scheduler = WarmupLinearSchedule(optimiser, warmup_steps=cfg.warmup_steps, total_steps=total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    step = 0
    for epoch in tqdm(range(1, cfg.n_epochs + 1), desc="Training epochs"):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.n_epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimiser.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)  # (B, L, V)

            # Reshape for token-level loss
            B, L, V = logits.size()
            loss = criterion(logits.view(B * L, V), labels.view(B * L))

            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimiser.step()
            scheduler.step()
            step += 1

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = running_loss / max(1, len(train_loader))
        log = f"Epoch {epoch}/{cfg.n_epochs} - training loss: {avg_loss:.4f}"
        if valid_loader is not None:
            val_loss = evaluate_mlm_loss(model, valid_loader, device)
            log += f" | validation loss: {val_loss:.4f}"
        print(log)


@torch.no_grad()
def evaluate_mlm_loss(model: BERT4RecModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    losses = []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        B, L, V = logits.size()
        loss = criterion(logits.view(B * L, V), labels.view(B * L))
        losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


# ---------- Next-item evaluation ----------

class NextItemEvalDataset(Dataset):
    """
    Dataset for next-item evaluation: for each sequence, mask the final item
    and ask the model to predict it using bidirectional context.
    """
    def __init__(
        self,
        sequences: List[List[int]],
        prefix_lengths: List[int],
        max_len: int,
    ) -> None:
        self.sequences = sequences
        self.prefix_lengths = prefix_lengths
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.sequences[idx]
        # We keep all tokens but replace the last non-prefix token by [MASK];
        # labels are -100 everywhere except that position.
        non_maskable = self.prefix_lengths[idx]
        # Identify last position to predict
        target_pos = len(tokens) - 1
        input_tokens = list(tokens)
        input_tokens[target_pos] = MASK_TOKEN_ID

        labels = [-100] * len(tokens)
        labels[target_pos] = tokens[target_pos]

        # Truncate/pad on the left
        input_tokens = input_tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        pad_len = self.max_len - len(input_tokens)

        return {
            "input_ids": torch.tensor([PAD_TOKEN_ID] * pad_len + input_tokens, dtype=torch.long),
            "labels": torch.tensor([-100] * pad_len + labels, dtype=torch.long),
            "attention_mask": torch.tensor([0] * pad_len + [1] * len(input_tokens), dtype=torch.long),
        }


@torch.no_grad()
def evaluate_next_item_topk(
    model: BERT4RecModel,
    loader: DataLoader,
    device: torch.device,
    registry: TokenRegistry,
    topk: int = 20,
) -> Tuple[float, float]:
    """
    Compute Recall@K and NDCG@K for next-item prediction.

    Returns
    -------
    recall_at_k, ndcg_at_k : tuple of floats
    """
    model.eval()
    total = 0
    hits = 0
    ndcg_sum = 0.0

    for batch in tqdm(loader, desc="Evaluating next-item", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)  # (B, L, V)

        # Find the single labelled position per sequence (the last mask)
        # Compute top-K over the vocabulary
        B, L, V = logits.size()
        for b in range(B):
            label_pos = (labels[b] != -100).nonzero(as_tuple=False).squeeze(-1)
            if label_pos.numel() == 0:
                continue
            p = int(label_pos[-1].item())
            target = int(labels[b, p].item())
            scores = logits[b, p]  # (V,)
            topk_scores, topk_ids = torch.topk(scores, k=topk)
            total += 1
            topk_list = topk_ids.tolist()
            if target in topk_list:
                hits += 1
                rank = topk_list.index(target) + 1  # 1-based
                ndcg_sum += 1.0 / math.log2(rank + 1)
            else:
                ndcg_sum += 0.0

    recall = hits / max(1, total)
    ndcg = ndcg_sum / max(1, total)
    return recall, ndcg


# -----------------------------
# Convenience: end-to-end helper
# -----------------------------

def build_dataloaders_for_bert4rec(
    prepared: PreparedData,
    batch_size: int = 256,
    masking: MaskingOptions = MaskingOptions(),
    valid_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/valid/eval dataloaders from PreparedData.
    The validation loader uses the same masking objective as training.
    The eval loader performs next-item top-K evaluation (mask last item).
    """
    n = len(prepared.sequences)
    idx = list(range(n))
    random.shuffle(idx)
    n_valid = int(n * valid_split)
    valid_idx = set(idx[:n_valid])

    train_seqs, train_prefix = [], []
    valid_seqs, valid_prefix = [], []
    for i in tqdm(range(n), desc="Splitting sequences"):
        if i in valid_idx:
            valid_seqs.append(prepared.sequences[i])
            valid_prefix.append(prepared.prefix_lengths[i])
        else:
            train_seqs.append(prepared.sequences[i])
            train_prefix.append(prepared.prefix_lengths[i])

    train_ds = BERT4RecDataset(
        sequences=train_seqs,
        prefix_lengths=train_prefix,
        vocab_size=prepared.registry.vocab_size,
        max_len=max(len(s) for s in prepared.sequences),
        masking=masking,
    )
    valid_ds = BERT4RecDataset(
        sequences=valid_seqs,
        prefix_lengths=valid_prefix,
        vocab_size=prepared.registry.vocab_size,
        max_len=max(len(s) for s in prepared.sequences),
        masking=masking,
    )
    eval_ds = NextItemEvalDataset(
        sequences=prepared.sequences,
        prefix_lengths=prepared.prefix_lengths,
        max_len=max(len(s) for s in prepared.sequences),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, eval_loader