"""
Training script for crossing-intention prediction.

Trains an IntentionLSTM on PIE bbox sequences, evaluating accuracy and F1
on val/test splits.

Usage:
    python -m training.train_intention
    python -m training.train_intention --obs-len 15 --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import PIEIntentionDataset
from models.lstm import IntentionLSTM


def get_class_weights(dataset: PIEIntentionDataset, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced data."""
    labels = [dataset.windows[i][1] for i in range(len(dataset))]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_neg = len(labels) / (2.0 * n_neg)
    w_pos = len(labels) / (2.0 * n_pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run model on a dataloader and return loss, accuracy, f1."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        bbox = batch["bbox"].to(device)
        label = batch["label"].to(device)
        logits = model(bbox)
        total_loss += criterion(logits, label).item()
        n_batches += 1
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(label.cpu().tolist())

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
    }


def train(
    pie_path: str = "data/pie",
    output_dir: str = "results/intention",
    obs_len: int = 15,
    stride: int = 5,
    batch_size: int = 64,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 50,
    patience: int = 10,
    device: str = "auto",
):
    """Full training loop with early stopping on val F1."""

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = PIEIntentionDataset(split="train", obs_len=obs_len, stride=stride, pie_path=pie_path)
    val_ds   = PIEIntentionDataset(split="val",   obs_len=obs_len, stride=stride, pie_path=pie_path)
    test_ds  = PIEIntentionDataset(split="test",  obs_len=obs_len, stride=stride, pie_path=pie_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    input_dim = train_ds[0]["bbox"].shape[-1]
    model = IntentionLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    # ── Training setup ────────────────────────────────────────────────
    class_weights = get_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_ckpt = out / "lstm_best.pt"

    # ── Training loop ─────────────────────────────────────────────────
    best_val_f1 = 0.0
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            bbox  = batch["bbox"].to(device)
            label = batch["label"].to(device)

            logits = model(bbox)
            loss   = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        val_metrics    = evaluate(model, val_loader, device)
        elapsed        = time.time() - t0

        record = {
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss":   round(val_metrics["loss"], 4),
            "val_acc":    round(val_metrics["accuracy"], 4),
            "val_f1":     round(val_metrics["f1"], 4),
            "lr":         round(scheduler.get_last_lr()[0], 6),
            "time":       round(elapsed, 1),
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_acc={record['val_acc']:.4f} | "
            f"val_f1={record['val_f1']:.4f} | "
            f"{record['time']:.1f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_ckpt)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ── Test evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1:       {test_metrics['f1']:.4f}")
    print(f"  Loss:     {test_metrics['loss']:.4f}")
    print(f"{'='*50}")

    results = {
        "input_dim":    input_dim,
        "obs_len":      obs_len,
        "hidden_dim":   hidden_dim,
        "num_layers":   num_layers,
        "n_params":     n_params,
        "best_val_f1":  round(best_val_f1, 4),
        "test":         {k: round(v, 4) for k, v in test_metrics.items()},
        "history":      history,
    }
    results_path = out / "lstm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Crossing-intention LSTM training")
    parser.add_argument("--pie-path",   default="data/pie")
    parser.add_argument("--output-dir", default="results/intention")
    parser.add_argument("--obs-len",    type=int,   default=15)
    parser.add_argument("--stride",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--num-layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
