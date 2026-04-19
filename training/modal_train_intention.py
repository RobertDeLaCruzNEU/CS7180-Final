"""
Modal wrapper for crossing-intention LSTM training on remote GPU.

Mounts the local source code and PIE data into the container, runs training,
and saves results to a Modal Volume. The best checkpoint is also written
locally to weights/ when training completes.

Mirrors the pattern from prev_CS_6170/src/scripts/experiment1/modal_train.py,
adapted for this project's flat layout (data/, models/, training/ at root).

Usage:
    modal run training/modal_train_intention.py
    modal run training/modal_train_intention.py --epochs 50 --obs-len 15
"""

import modal
from pathlib import Path

app = modal.App("pie-crossing-intention")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "pydantic",
    )
    # Mount source modules so imports work inside the container
    .add_local_file("data/pie_data.py",  remote_path="/root/project/data/pie_data.py")
    .add_local_file("data/__init__.py",  remote_path="/root/project/data/__init__.py")
    .add_local_file("data/dataset.py",   remote_path="/root/project/data/dataset.py")
    .add_local_dir("models",             remote_path="/root/project/models")
    .add_local_dir("training",           remote_path="/root/project/training")
    # PIE class needs annotation dirs to discover set IDs even when loading
    # from the cached pickle (same requirement as prev_CS_6170)
    .add_local_dir("data/pie/data_cache",              remote_path="/root/project/data/pie/data_cache")
    .add_local_dir("data/pie/annotations",             remote_path="/root/project/data/pie/annotations")
    .add_local_dir("data/pie/annotations_attributes",  remote_path="/root/project/data/pie/annotations_attributes")
    .add_local_dir("data/pie/annotations_vehicle",     remote_path="/root/project/data/pie/annotations_vehicle")
)

# Persistent volume for saving results and checkpoints
results_vol = modal.Volume.from_name("pie-results", create_if_missing=True)

_WEIGHTS_FILENAME = "intention_lstm_best.pt"


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/root/results": results_vol},
)
def train_model(epochs: int = 50, obs_len: int = 15):
    import sys

    sys.path.insert(0, "/root/project")

    from training.train_intention import train

    output_dir = f"/root/results/intention/{obs_len}f"

    results = train(
        pie_path="/root/project/data/pie",
        output_dir=output_dir,
        obs_len=obs_len,
        epochs=epochs,
        device="auto",
    )

    results_vol.commit()

    # Return checkpoint bytes so the local entrypoint can write them to weights/
    ckpt_path = Path(output_dir) / "lstm_best.pt"
    ckpt_bytes = ckpt_path.read_bytes()

    return results, ckpt_bytes


def _save_weights(ckpt_bytes: bytes, dest: str = _WEIGHTS_FILENAME) -> None:
    out = Path("weights") / dest
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(ckpt_bytes)
    print(f"Weights saved locally → {out}")


@app.local_entrypoint()
def main(epochs: int = 50, obs_len: int = 15):
    print("=" * 60)
    print(f"Training crossing-intention LSTM (obs_len={obs_len}f, epochs={epochs})")
    print("=" * 60)

    results, ckpt_bytes = train_model.remote(epochs=epochs, obs_len=obs_len)

    print(f"\nTEST RESULTS")
    print(f"  Accuracy: {results['test']['accuracy']:.4f}")
    print(f"  F1:       {results['test']['f1']:.4f}")
    print(f"  Loss:     {results['test']['loss']:.4f}")

    _save_weights(ckpt_bytes)
