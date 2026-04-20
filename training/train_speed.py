"""
Training script for ego vehicle speed prediction.

Trains a SpeedLSTM on PIE bbox + crossing-intention sequences, regressing
against the ground truth OBD speed at the last frame of each window.

Metrics are reported in km/h (predictions rescaled from [0,1] by MAX_SPEED_KMH).

Usage:
    python -m training.train_speed
    python -m training.train_speed --obs-len 15 --epochs 50
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import PIESpeedDataset, MAX_SPEED_KMH
from models.speed_lstm import SpeedLSTM


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run model on a dataloader and return MSE loss, MAE and RMSE in km/h."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    abs_errors = []
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        target   = batch["speed"].to(device)           # normalised [0, 1]

        pred = model(features)
        total_loss += criterion(pred, target).item()
        n_batches  += 1

        # Convert to km/h for interpretable metrics
        pred_kmh   = pred.cpu() * MAX_SPEED_KMH
        target_kmh = target.cpu() * MAX_SPEED_KMH
        abs_errors.append((pred_kmh - target_kmh).abs())

    all_errors = torch.cat(abs_errors)
    mae  = all_errors.mean().item()
    rmse = math.sqrt((all_errors ** 2).mean().item())

    return {
        "loss": total_loss / max(n_batches, 1),
        "mae_kmh":  mae,
        "rmse_kmh": rmse,
    }


def train(
    pie_path:   str   = "data/pie",
    output_dir: str   = "results/speed",
    obs_len:    int   = 15,
    stride:     int   = 5,
    batch_size: int   = 64,
    hidden_dim: int   = 64,
    num_layers: int   = 2,
    dropout:    float = 0.1,
    lr:         float = 1e-3,
    epochs:     int   = 50,
    patience:   int   = 10,
    device:     str   = "auto",
):
    """Full training loop with early stopping on val MAE."""

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = PIESpeedDataset(split="train", obs_len=obs_len, stride=stride, pie_path=pie_path)
    val_ds   = PIESpeedDataset(split="val",   obs_len=obs_len, stride=stride, pie_path=pie_path)
    test_ds  = PIESpeedDataset(split="test",  obs_len=obs_len, stride=stride, pie_path=pie_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────
    input_dim = train_ds[0]["features"].shape[-1]   # 5
    model = SpeedLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}")

    # ── Training setup ────────────────────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_ckpt = out / "lstm_best.pt"

    # ── Training loop ─────────────────────────────────────────────────
    best_val_mae = float("inf")
    wait    = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)
            target   = batch["speed"].to(device)

            pred = model(features)
            loss = criterion(pred, target)

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
            "epoch":       epoch,
            "train_loss":  round(avg_train_loss, 6),
            "val_loss":    round(val_metrics["loss"], 6),
            "val_mae_kmh": round(val_metrics["mae_kmh"], 4),
            "val_rmse_kmh":round(val_metrics["rmse_kmh"], 4),
            "lr":          round(scheduler.get_last_lr()[0], 6),
            "time":        round(elapsed, 1),
        }
        history.append(record)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train={record['train_loss']:.6f} | "
            f"val_loss={record['val_loss']:.6f} | "
            f"val_mae={record['val_mae_kmh']:.2f} km/h | "
            f"val_rmse={record['val_rmse_kmh']:.2f} km/h | "
            f"{record['time']:.1f}s"
        )

        if val_metrics["mae_kmh"] < best_val_mae:
            best_val_mae = val_metrics["mae_kmh"]
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
    print(f"  MAE:  {test_metrics['mae_kmh']:.4f} km/h")
    print(f"  RMSE: {test_metrics['rmse_kmh']:.4f} km/h")
    print(f"  Loss: {test_metrics['loss']:.6f}")
    print(f"{'='*50}")

    results = {
        "input_dim":    input_dim,
        "obs_len":      obs_len,
        "hidden_dim":   hidden_dim,
        "num_layers":   num_layers,
        "n_params":     n_params,
        "best_val_mae": round(best_val_mae, 4),
        "test":         {k: round(v, 4) for k, v in test_metrics.items()},
        "history":      history,
    }
    results_path = out / "lstm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Speed prediction LSTM training")
    parser.add_argument("--pie-path",   default="data/pie")
    parser.add_argument("--output-dir", default="results/speed")
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
