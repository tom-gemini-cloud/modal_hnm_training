#!/bin/bash

# run_training.sh
# Script to launch BERT4Rec training on Modal with different configurations

# Default configuration
EXPERIMENT_NAME="bert4rec_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
BATCH_SIZE=256
LEARNING_RATE=0.001
GPU_TYPE="L4"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --gpu)
      GPU_TYPE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run_training.sh [options]"
      echo "Options:"
      echo "  --name NAME          Experiment name (default: bert4rec_TIMESTAMP)"
      echo "  --epochs N           Number of epochs (default: 10)"
      echo "  --batch-size N       Batch size (default: 256)"
      echo "  --lr RATE           Learning rate (default: 0.001)"
  echo "  --gpu TYPE          GPU type: L4, A10G, A100, or A100-80GB (default: L4)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "BERT4Rec Training on Modal"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "GPU Type: $GPU_TYPE"
echo "=========================================="

# Run the training job on Modal
modal run --env hnmdev modal/train_modal.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --learning-rate $LEARNING_RATE \
  --gpu-type "$GPU_TYPE"

echo "Training job submitted to Modal!"