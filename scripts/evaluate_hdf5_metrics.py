import argparse
import h5py
import numpy as np
from sklearn.metrics import r2_score


def compute_metrics(pred, target):
    diff = pred.reshape(-1) - target.reshape(-1)
    abs_err = np.abs(diff)
    return {
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "MAE": float(np.mean(abs_err)),
        "R2": float(r2_score(target.reshape(-1), pred.reshape(-1))),
        "P90_AE": float(np.percentile(abs_err, 90)),
        "P95_AE": float(np.percentile(abs_err, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HARP-FiLM HDF5 prediction output.")
    parser.add_argument("--input", required=True, help="HDF5 file containing prediction and target datasets.")
    args = parser.parse_args()
    with h5py.File(args.input, "r") as h5f:
        pred = h5f["prediction"][:].astype(np.float64)
        target = h5f["target"][:].astype(np.float64)
    metrics = compute_metrics(pred, target)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
