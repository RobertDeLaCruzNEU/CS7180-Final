"""
Modal wrapper for speed prediction LSTM training on remote GPU.

Mounts the local source code and PIE data into the container, runs training,
and saves results to a Modal Volume. The best checkpoint is also written
locally to weights/ when training completes.

Mirrors training/modal_train_intention.py exactly, adapted for speed regression.

Usage:
    modal run training/modal_train_speed.py
    modal run training/modal_train_speed.py --epochs 50 --obs-len 15
"""

import modal
from pathlib import Path

app = modal.App("pie-speed-prediction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "scikit-learn",
        "pydantic",
    )
    .add_local_file("data/pie_data.py",  remote_path="/root/project/data/pie_data.py")
    .add_local_file("data/__init__.py",  remote_path="/root/project/data/__init__.py")
    .add_local_file("data/dataset.py",   remote_path="/root/project/data/dataset.py")
    .add_local_dir("models",             remote_path="/root/project/models")
    .add_local_dir("training",           remote_path="/root/project/training")
    .add_local_dir("data/pie/data_cache",             remote_path="/root/project/data/pie/data_cache")
    .add_local_dir("data/pie/annotations",            remote_path="/root/project/data/pie/annotations")
    .add_local_dir("data/pie/annotations_attributes", remote_path="/root/project/data/pie/annotations_attributes")
    .add_local_dir("data/pie/annotations_vehicle",    remote_path="/root/project/data/pie/annotations_vehicle")
)

results_vol = modal.Volume.from_name("pie-results", create_if_missing=True)

_WEIGHTS_FILENAME = "speed_lstm_best.pt"


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={"/root/results": results_vol},
)
def train_model(epochs: int = 50, obs_len: int = 15):
    import sys

    sys.path.insert(0, "/root/project")

    from training.train_speed import train

    output_dir = f"/root/results/speed/{obs_len}f"

    results = train(
        pie_path="/root/project/data/pie",
        output_dir=output_dir,
        obs_len=obs_len,
        epochs=epochs,
        device="auto",
    )

    results_vol.commit()

    ckpt_path  = Path(output_dir) / "lstm_best.pt"
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
    print(f"Training speed prediction LSTM (obs_len={obs_len}f, epochs={epochs})")
    print("=" * 60)

    results, ckpt_bytes = train_model.remote(epochs=epochs, obs_len=obs_len)

    print(f"\nTEST RESULTS")
    print(f"  MAE:  {results['test']['mae_kmh']:.4f} km/h")
    print(f"  RMSE: {results['test']['rmse_kmh']:.4f} km/h")
    print(f"  Loss: {results['test']['loss']:.6f}")

    _save_weights(ckpt_bytes)
