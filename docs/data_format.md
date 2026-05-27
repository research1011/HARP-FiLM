# Expected HDF5 data format

This repository expects preprocessed HDF5 files that combine CODE GIM VTEC maps and OMNI2 solar–geomagnetic drivers on an hourly timeline.

## File-level requirement

Each split is stored as one HDF5 file:

| File | Period | Purpose |
|---|---:|---|
| `train_aux_0817_single.h5` | 2008–2017 | training |
| `val_aux_1819_single.h5` | 2018–2019 | validation |
| `test_aux_2020_single.h5` | 2020 | independent low-activity test |
| `test_aux_2023_single.h5` | 2023 | independent elevated-activity test |

## Required core datasets

Each HDF5 file must contain the following keys.

| Key | Accepted shape | Dtype | Description |
|---|---:|---|---|
| `Timestamp` | `(T,)` | byte/string | ISO timestamps, e.g., `2023-01-01T00:00:00` |
| `TEC` | `(T, 71, 73)` or `(T, 5183)` | float | global VTEC maps in TECU |
| `Latitude` | `(T, 71, 73)` or `(T, 5183)` | float | latitude grid, repeated over time if needed |
| `Longitude` | `(T, 71, 73)` or `(T, 5183)` | float | longitude grid, repeated over time if needed |

The model assumes a CODE GIM grid of 71 × 73, corresponding to latitudes from 87.5°S to 87.5°N and longitudes from 180°W to 180°E.

## Auxiliary-driver datasets

The manuscript uses five selected solar–geomagnetic drivers. Each driver should be stored as a separate one-dimensional HDF5 dataset of length `T`.

| HDF5 key used in config | Accepted shape | Meaning |
|---|---:|---|
| `f10.7_index` | `(T,)` or `(T, 1)` | F10.7 solar radio flux index |
| `AU-index` | `(T,)` or `(T, 1)` | auroral upper index |
| `DST Index` | `(T,)` or `(T, 1)` | Dst geomagnetic disturbance index |
| `Plasma (Flow) speed` | `(T,)` or `(T, 1)` | solar-wind proton flow speed |
| `Na/Np` | `(T,)` or `(T, 1)` | alpha-to-proton density ratio |

If your HDF5 files use a different key name, modify `selected_aux_keys` in `configs/harp_film.yaml`. For example, some OMNI-derived files may use `R` for the solar radio flux index. To keep the implementation aligned with the manuscript, the key mapping should be documented before release.

## Time encodings

Four temporal features are generated internally from `Timestamp` and should not be stored in the HDF5 file:

- `hour_sin`
- `hour_cos`
- `doy_sin`
- `doy_cos`

Thus, the model receives 9 tabular features at each input time step: 5 physical drivers + 4 temporal encodings.

## Missing and invalid values

Before creating the processed HDF5 files, invalid OMNI-style fill values should be handled consistently. For the auxiliary drivers used in this study, invalid values such as `9.999` and `999.9` should be replaced by linear interpolation along the time dimension for each variable separately. Do not interpolate across different variables within the same row.

The training code also applies `np.nan_to_num` as a final safety guard, but the recommended practice is to clean invalid values before writing the processed HDF5 files.

## Preprocessing used during training

The training script applies the following transformations:

1. TEC target/input values: `log1p(TEC)` followed by `StandardScaler` fitted only on training historical frames.
2. Auxiliary drivers: `MinMaxScaler` fitted only on training historical frames.
3. Time encodings: sine/cosine features generated from timestamps and not scaled.
4. Latitude/longitude channels: sine/cosine encodings appended to each input VTEC map.

During evaluation, predictions and targets are transformed back to original TECU scale before metrics are computed.
