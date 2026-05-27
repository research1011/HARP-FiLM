# Preprocessing notes

This document summarizes the preprocessing assumed by the released HARP-FiLM code.

## 1. CODE GIM processing

The target variable is the global ionospheric vertical total electron content (VTEC) from CODE Global Ionosphere Maps.

- Spatial grid: 71 × 73.
- Unit: TECU.
- Native 1 h products are retained directly.
- Native 2 h products can be linearly interpolated along the time dimension to construct a consistent hourly sequence.
- No additional spatial interpolation is required beyond the original GIM grid.

## 2. OMNI2 driver processing

The external physical drivers are aligned to the same hourly timestamps as the VTEC maps.

The final selected driver set used by the manuscript is:

1. F10.7 index
2. AU index
3. Dst index
4. solar-wind proton flow speed, Vp
5. alpha-to-proton density ratio, Na/Np

If raw OMNI2 files contain fill values or invalid markers, they should be converted to `NaN` and interpolated along time for each driver column separately.

## 3. Processed HDF5 assembly

After VTEC and OMNI2 drivers are aligned, each chronological split should be written as one HDF5 file with the key structure described in `data_format.md`.

No training/validation/test statistics should be fitted during this HDF5 assembly step. Scaling is performed by the training script using only the training split.

## 4. Leakage avoidance

The released training code fits:

- the TEC `StandardScaler` only on `log1p(TEC)` values from the training historical input frames;
- the auxiliary-driver `MinMaxScaler` only on selected auxiliary-driver values from the training historical input frames.

Validation and test splits are transformed using these training-fitted scalers.
