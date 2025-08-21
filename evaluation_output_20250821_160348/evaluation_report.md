# Model Evaluation Results

**Experiment:** bert4rec_20250821_121512
**Best Epoch:** 20
**MLM Loss:** 6.3208

## Accuracy Metrics
- **accuracy@1:** 0.0832 (8.32%)

## Precision@K
- **P@5:** 0.0341 (3.41%)
- **P@10:** 0.0219 (2.19%)
- **P@20:** 0.0136 (1.36%)
- **P@50:** 0.0071 (0.71%)

## Recall@K
- **R@20:** 0.2719 (27.19%)
- **R@5:** 0.1705 (17.05%)
- **R@10:** 0.2190 (21.90%)
- **R@50:** 0.3574 (35.74%)

## F1-Score@K
- **F1@5:** 0.0568 (5.68%)
- **F1@10:** 0.0398 (3.98%)
- **F1@20:** 0.0259 (2.59%)
- **F1@50:** 0.0140 (1.40%)

## NDCG@K
- **NDCG@20:** 0.1577 (15.77%)
- **NDCG@5:** 0.1286 (12.86%)
- **NDCG@10:** 0.1443 (14.43%)
- **NDCG@50:** 0.1746 (17.46%)

## Summary Table

| K | Precision | Recall | F1-Score | NDCG |
|---|-----------|--------|----------|------|
| 5 | 0.0341 | 0.1705 | 0.0568 | 0.1286 |
| 10 | 0.0219 | 0.2190 | 0.0398 | 0.1443 |
| 20 | 0.0136 | 0.2719 | 0.0259 | 0.1577 |
| 50 | 0.0071 | 0.3574 | 0.0140 | 0.1746 |

## Model Configuration

### Model Parameters
- **d_model:** 256
- **n_heads:** 4
- **n_layers:** 3
- **dim_feedforward:** 512
- **max_len:** 100
- **dropout:** 0.1
- **vocab_size:** 42300

### Training Parameters
- **batch_size:** 512
- **learning_rate:** 0.001
- **weight_decay:** 0.0001
- **n_epochs:** 20
- **warmup_steps:** 1000
- **grad_clip_norm:** 1.0

### Data Parameters
- **train_size:** 295613
- **valid_size:** 36951
- **test_size:** 36953
- **min_seq_len:** 3
- **mask_prob:** 0.15