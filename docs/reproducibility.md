# Reproducibility guide

## Main chronological split

The main experiments use the following split:

| Split | Period |
|---|---:|
| Training | 2008–2017 |
| Validation | 2018–2019 |
| Test | 2020 and 2023 |

The default config tests 2023. To evaluate 2020, change:

```yaml
data:
  test_hdf5_path: "data/test_aux_2020_single.h5"
```

## Sliding-window construction

The main forecasting setting is:

- input length: 24 h
- prediction horizon: 24 h
- training stride: 6 h
- validation/test stride: 24 h

A candidate window is retained only if all timestamps inside the 48 h input-output window are continuous at the inferred dominant time step.

## Model hyperparameters

The default model configuration is:

| Hyperparameter | Value |
|---|---:|
| latent dimension `D` | 256 |
| Transformer heads | 8 |
| Transformer layers | 4 |
| feed-forward dimension | 256 |
| dropout | 0.15 |
| CNN channels | 12 |
| attention pooling recent steps | 8 |
| prediction horizon | 24 |

## Optimization

The default optimization setting is:

| Setting | Value |
|---|---:|
| loss | SmoothL1 |
| optimizer | AdamW |
| learning rate | 2e-4 |
| weight decay | 1e-5 |
| scheduler | ReduceLROnPlateau |
| scheduler factor | 0.3 |
| scheduler patience | 10 |
| minimum learning rate | 1e-6 |
| early stopping patience | 20 |
| gradient clipping | 0.5 |
| random seed | 42 |

## Blocked chronological stress test

For the solar-maximum stress-test setting described in the manuscript, retrain the model using only 2008–2013 and then test on 2014 and 2017. This requires preparing corresponding HDF5 split files and updating `configs/harp_film.yaml` accordingly.

Example:

```yaml
data:
  train_hdf5_path: "data/train_aux_0813_single.h5"
  val_hdf5_path: "data/val_aux_selected_single.h5"
  test_hdf5_path: "data/test_aux_2014_single.h5"
```

The exact validation split for the blocked setting should follow the experimental design used in your manuscript records.

## Interpretability outputs

The core model can optionally return internal diagnostic variables:

```python
y, diagnostics = model(x_dec, x_mark_dec, return_diagnostics=True)
```

The diagnostic dictionary contains:

- `temporal_attention`: horizon-aware attention weights over recent hidden states;
- `film.gamma_tanh`, `film.beta`, `film.gate`: Physics-FiLM modulation variables;
- `residual_head.alpha`, `residual_head.delta`, `residual_head.gain`, `residual_head.bias`: dual-baseline residual variables.

These variables can be used to reproduce interpretability analyses such as temporal attention maps, FiLM statistics, dual-baseline fusion behavior, and driver-response correlations.
