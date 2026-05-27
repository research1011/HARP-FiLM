import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

TIME_KEYS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
CORE_KEYS = ["Timestamp", "Latitude", "Longitude", "TEC"]


def _decode_timestamps(ts_bytes) -> pd.DatetimeIndex:
    ts_list = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t) for t in ts_bytes]
    return pd.to_datetime(ts_list, format="%Y-%m-%dT%H:%M:%S", errors="raise")


def load_timestamps(hdf5_path: str) -> pd.DatetimeIndex:
    with h5py.File(hdf5_path, "r") as h5f:
        return _decode_timestamps(h5f["Timestamp"][:])


def infer_step(timestamps_np) -> np.timedelta64:
    diffs = np.diff(timestamps_np)
    secs = diffs.astype("timedelta64[s]").astype(np.int64)
    if len(secs) == 0:
        return np.timedelta64(3600, "s")
    values, counts = np.unique(secs, return_counts=True)
    return np.timedelta64(int(values[np.argmax(counts)]), "s")


def build_valid_indices(timestamps_pd: pd.DatetimeIndex, input_len: int, output_len: int, stride: int, expected_step) -> List[int]:
    ts = timestamps_pd.to_numpy(dtype="datetime64[s]")
    win = input_len + output_len
    indices = []
    for s in range(0, len(ts) - win + 1, stride):
        sub = ts[s : s + win]
        if np.all(np.diff(sub) == expected_step):
            indices.append(s)
    return indices


def read_aux_keys(hdf5_path: str, selected_aux_keys: Optional[Sequence[str]] = None) -> List[str]:
    with h5py.File(hdf5_path, "r") as h5f:
        file_aux_all = [k for k in h5f.keys() if k not in CORE_KEYS]
    if selected_aux_keys:
        missing = [k for k in selected_aux_keys if k not in file_aux_all]
        if missing:
            raise ValueError(f"Missing selected auxiliary keys in {hdf5_path}: {missing}")
        return list(selected_aux_keys)
    return sorted(file_aux_all)


def make_train_history_mask(T_train: int, train_indices: Sequence[int], input_timestamps: int) -> np.ndarray:
    mask = np.zeros(T_train, dtype=bool)
    for s in train_indices:
        mask[s : s + input_timestamps] = True
    if not np.any(mask):
        raise RuntimeError("No training history frames available for scaler fitting.")
    return mask


def fit_tec_scaler(train_hdf5_path: str, train_indices: Sequence[int], input_timestamps: int, H: int = 71, W: int = 73) -> StandardScaler:
    with h5py.File(train_hdf5_path, "r") as h5f:
        T_train = h5f["Timestamp"].shape[0]
        tec = h5f["TEC"][:].reshape(T_train, H, W)
    tec = np.nan_to_num(tec, nan=0.0, posinf=0.0, neginf=0.0)
    mask = make_train_history_mask(T_train, train_indices, input_timestamps)
    tec_train_log = np.log1p(tec[mask]).reshape(-1, 1)
    scaler = StandardScaler().fit(tec_train_log)
    logger.info("Fitted TEC scaler using log1p(TEC) on training historical frames only.")
    return scaler


def fit_aux_scaler(train_hdf5_path: str, aux_keys: Sequence[str], train_indices: Sequence[int], input_timestamps: int) -> Optional[MinMaxScaler]:
    if not aux_keys:
        return None
    with h5py.File(train_hdf5_path, "r") as h5f:
        T_train = h5f["Timestamp"].shape[0]
        arrays = []
        for k in aux_keys:
            a = h5f[k][:]
            if a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if a.shape[0] != T_train:
                raise ValueError(f"Auxiliary feature {k} length {a.shape[0]} differs from timestamp length {T_train}.")
            arrays.append(a)
    aux_features = np.column_stack(arrays).astype(np.float32)
    mask = make_train_history_mask(T_train, train_indices, input_timestamps)
    scaler = MinMaxScaler().fit(aux_features[mask])
    logger.info("Fitted auxiliary MinMax scaler on training historical frames only.")
    return scaler


class TECPredictDataset(Dataset):
    """Dataset for 24 h input to 24 h output global VTEC forecasting.

    TEC preprocessing follows the manuscript: log1p transformation followed by a StandardScaler
    fitted only on training historical frames. Auxiliary drivers use a MinMaxScaler fitted only
    on training historical frames. Temporal sine/cosine encodings are appended without scaling.
    """

    def __init__(
        self,
        fused_hdf5_path: str,
        input_timestamps: int,
        output_timestamps: int,
        indices: Sequence[int],
        tec_scaler: StandardScaler,
        fixed_aux_keys: Optional[Sequence[str]] = None,
        aux_scaler: Optional[MinMaxScaler] = None,
        num_latitudes: int = 71,
        num_longitudes: int = 73,
    ):
        self.fused_hdf5_path = str(fused_hdf5_path)
        self.input_timestamps = int(input_timestamps)
        self.output_timestamps = int(output_timestamps)
        self.indices = list(indices)
        self.scaler = tec_scaler
        self.aux_scaler = aux_scaler
        self.num_latitudes = int(num_latitudes)
        self.num_longitudes = int(num_longitudes)

        with h5py.File(self.fused_hdf5_path, "r") as h5f:
            self.timestamps = _decode_timestamps(h5f["Timestamp"][:])
            T = len(self.timestamps)
            tec = h5f["TEC"][:].reshape(T, self.num_latitudes, self.num_longitudes)
            self.tec = np.nan_to_num(tec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            lat_all = h5f["Latitude"][:]
            lon_all = h5f["Longitude"][:]
            self.lat_grid = lat_all.reshape(T, self.num_latitudes, self.num_longitudes)[0].astype(np.float32)
            self.lon_grid = lon_all.reshape(T, self.num_latitudes, self.num_longitudes)[0].astype(np.float32)

            if self.indices:
                max_idx = max(self.indices)
                win = self.input_timestamps + self.output_timestamps
                if max_idx + win > T:
                    raise ValueError(f"Index out of bounds: {max_idx} + {win} > {T}")

            self.aux_keys = read_aux_keys(self.fused_hdf5_path, fixed_aux_keys)
            time_feats = self._build_time_features(self.timestamps)

            if self.aux_keys:
                aux_arrays = []
                for k in self.aux_keys:
                    a = h5f[k][:]
                    if a.ndim == 2 and a.shape[1] == 1:
                        a = a[:, 0]
                    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    if a.shape[0] != T:
                        raise ValueError(f"Auxiliary feature {k} length {a.shape[0]} differs from timestamp length {T}.")
                    aux_arrays.append(a)
                aux_features = np.column_stack(aux_arrays).astype(np.float32)
                if self.aux_scaler is not None:
                    aux_features = self.aux_scaler.transform(aux_features)
                self.combined_features = np.concatenate([aux_features, time_feats], axis=1).astype(np.float32)
            else:
                self.combined_features = time_feats.astype(np.float32)

        lat_rad = np.deg2rad(self.lat_grid)
        lon_rad = np.deg2rad(self.lon_grid)
        base_geo = np.stack(
            [
                np.sin(lat_rad).astype(np.float32),
                np.cos(lat_rad).astype(np.float32),
                np.sin(lon_rad).astype(np.float32),
                np.cos(lon_rad).astype(np.float32),
            ],
            axis=0,
        )
        self.geo_repeat = np.repeat(base_geo[np.newaxis, ...], self.input_timestamps, axis=0).astype(np.float32)

    @staticmethod
    def _build_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
        hour = timestamps.hour.to_numpy()
        doy = timestamps.dayofyear.to_numpy()
        hour_rad = 2 * np.pi * (hour / 24.0)
        doy_rad = 2 * np.pi * ((doy - 1) / 366.0)
        return np.column_stack([np.sin(hour_rad), np.cos(hour_rad), np.sin(doy_rad), np.cos(doy_rad)]).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        mid_idx = start_idx + self.input_timestamps
        end_idx = mid_idx + self.output_timestamps

        inputs_tec = self.tec[start_idx:mid_idx]
        outputs = self.tec[mid_idx:end_idx]

        inputs_log = np.log1p(inputs_tec).reshape(-1, 1)
        outputs_log = np.log1p(outputs).reshape(-1, 1)
        inputs_tec_norm = self.scaler.transform(inputs_log).reshape(inputs_tec.shape).astype(np.float32)
        outputs_norm = self.scaler.transform(outputs_log).reshape(outputs.shape).astype(np.float32)

        inputs = np.concatenate([inputs_tec_norm[:, None, :, :], self.geo_repeat], axis=1).astype(np.float32)
        input_marks = self.combined_features[start_idx:mid_idx]
        output_marks = self.combined_features[mid_idx:end_idx]
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(outputs_norm),
            torch.from_numpy(input_marks),
            torch.from_numpy(output_marks),
        )


def worker_init_fn(worker_id: int):
    np.random.seed(42 + worker_id)


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )


def save_scalers(output_dir: str, tec_scaler, aux_scaler, aux_keys: Sequence[str]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tec_scaler, output_dir / "scaler_tec.joblib")
    if aux_scaler is not None:
        joblib.dump(aux_scaler, output_dir / "aux_scaler.joblib")
    with open(output_dir / "aux_keys.json", "w", encoding="utf-8") as f:
        import json

        json.dump(list(aux_keys), f, indent=2, ensure_ascii=False)
