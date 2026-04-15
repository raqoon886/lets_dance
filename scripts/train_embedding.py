"""
Train Dance Embedding Model
Trains the ST-GCN embedding model on collected dance data.

Usage:
    python scripts/train_embedding.py --data-dir data/reference_dances --epochs 100
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train dance embedding model")
    parser.add_argument("--data-dir", type=str, default="data/reference_dances",
                        help="Directory containing reference dance data")
    parser.add_argument("--output", type=str, default="data/models/dance_embedding.pth",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension")
    return parser.parse_args()


def train(data_dir: str, output_path: str, epochs: int, batch_size: int,
          lr: float, embedding_dim: int):
    """
    Train the dance embedding model.

    Training strategy:
        - Contrastive learning: same dance = similar embeddings,
          different dances = dissimilar embeddings
        - Triplet loss: (anchor, positive, negative) triplets
        - Data augmentation: time shift, speed variation, noise injection

    Pipeline:
        1. Load all reference dance landmark sequences
        2. Create training triplets
        3. Build ST-GCN model
        4. Train with triplet loss
        5. Save trained model weights
    """
    # TODO:
    # 1. Load landmark .npy files from data_dir subfolders
    # 2. Create DataLoader with triplet sampling
    # 3. Build DanceEmbeddingModel
    # 4. Training loop with loss tracking
    # 5. Save best model
    print(f"[MOCK] Training embedding model")
    print(f"  Data: {data_dir}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Output: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # TODO: actual training
    print("[MOCK] Training complete. Model saved.")


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.output, args.epochs, args.batch_size,
          args.lr, args.embedding_dim)
