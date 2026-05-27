import logging
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 20, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None
        self.early_stop = False

    def __call__(self, val_loss: float, model):
        score = -val_loss
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info("EarlyStopping counter: %s / %s", self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
            self.counter = 0


def inverse_tec_transform(x_norm: torch.Tensor, scaler_tec) -> torch.Tensor:
    mean = torch.tensor(float(scaler_tec.mean_[0]), device=x_norm.device, dtype=x_norm.dtype)
    std = torch.tensor(float(scaler_tec.scale_[0]), device=x_norm.device, dtype=x_norm.dtype)
    x_log = x_norm * std + mean
    return torch.expm1(x_log)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float):
    model.train()
    total_loss = 0.0
    for data, target, input_mark, _ in tqdm(loader, desc="Training", leave=False):
        x_dec = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        x_mark_dec = input_mark.to(device, non_blocking=True)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_dec, x_mark_dec)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for data, target, input_mark, _ in tqdm(loader, desc="Validation", leave=False):
        x_dec = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        x_mark_dec = input_mark.to(device, non_blocking=True)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        pred = model(x_dec, x_mark_dec)
        loss = criterion(pred, target)
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def predict_original_scale(model, loader, device, scaler_tec):
    model.eval()
    all_preds, all_targets = [], []
    for data, target, input_mark, _ in tqdm(loader, desc="Testing", leave=False):
        x_dec = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        x_mark_dec = input_mark.to(device, non_blocking=True)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
        pred = model(x_dec, x_mark_dec)
        all_preds.append(inverse_tec_transform(pred, scaler_tec).cpu().numpy())
        all_targets.append(inverse_tec_transform(target, scaler_tec).cpu().numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics(pred: np.ndarray, target: np.ndarray):
    diff = pred.reshape(-1) - target.reshape(-1)
    abs_err = np.abs(diff)
    return {
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "MAE": float(np.mean(abs_err)),
        "R2": float(r2_score(target.reshape(-1), pred.reshape(-1))),
        "P90_AE": float(np.percentile(abs_err, 90)),
        "P95_AE": float(np.percentile(abs_err, 95)),
    }


def save_results_to_hdf5(predictions, targets, test_indices, input_timestamps, timestamps, lat_grid, lon_grid, output_hdf5_path):
    output_hdf5_path = Path(output_hdf5_path)
    output_hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples, pred_len, lat, lon = predictions.shape
    total_entries = num_samples * pred_len * lat * lon
    sample_ids = np.zeros(total_entries, dtype=np.int32)
    iso_timestamps = np.empty(total_entries, dtype="S19")
    latitudes = np.zeros(total_entries, dtype=np.float32)
    longitudes = np.zeros(total_entries, dtype=np.float32)
    pred_values = np.zeros(total_entries, dtype=np.float32)
    target_values = np.zeros(total_entries, dtype=np.float32)
    idx = 0
    np_timestamps = np.array(timestamps, dtype="datetime64[s]")
    for s in range(num_samples):
        start = test_indices[s] + input_timestamps
        for t in range(pred_len):
            ts_str = np.datetime_as_string(np_timestamps[start + t], unit="s")
            for i in range(lat):
                for j in range(lon):
                    sample_ids[idx] = s
                    iso_timestamps[idx] = ts_str.encode("ascii")
                    latitudes[idx] = lat_grid[i, j]
                    longitudes[idx] = lon_grid[i, j]
                    pred_values[idx] = predictions[s, t, i, j]
                    target_values[idx] = targets[s, t, i, j]
                    idx += 1
    with h5py.File(output_hdf5_path, "w") as h5f:
        h5f.create_dataset("sample", data=sample_ids, compression="gzip", compression_opts=4)
        h5f.create_dataset("timestamp", data=iso_timestamps, compression="gzip", compression_opts=4)
        h5f.create_dataset("latitude", data=latitudes, compression="gzip", compression_opts=4)
        h5f.create_dataset("longitude", data=longitudes, compression="gzip", compression_opts=4)
        h5f.create_dataset("prediction", data=pred_values, compression="gzip", compression_opts=4)
        h5f.create_dataset("target", data=target_values, compression="gzip", compression_opts=4)
    logger.info("Saved prediction HDF5 to %s", output_hdf5_path)
