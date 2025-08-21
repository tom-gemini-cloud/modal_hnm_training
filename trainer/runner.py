# trainer/runner.py
import argparse
from trainer.bert4rec_modelling import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-uri", required=True)
    p.add_argument("--val-uri")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    train(
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )