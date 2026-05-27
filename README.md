# HARP-FiLM: Global Ionospheric VTEC Forecasting

This repository provides the PyTorch implementation of **HARP-FiLM** (*Horizon-Aware Residual Prediction with Physics-Guided Feature-wise Linear Modulation*) for 24 h global ionospheric vertical total electron content (VTEC) forecasting.

The repository is prepared for the manuscript:

> **HARP-FiLM: Global Ionospheric VTEC Forecasting with Physics-Guided Feature Modulation and Dual-Baseline Residuals**

HARP-FiLM combines multi-scale spatial encoding, physics-guided feature-wise modulation, Transformer-based temporal modeling, horizon-aware temporal attention pooling, and adaptive dual-baseline residual forecasting.

## Model overview

The implementation corresponds to the method described in the manuscript:

1. **Multi-scale CNN spatial encoder**: extracts global VTEC morphology using 3 × 3, 5 × 5, and 7 × 7 convolutional branches.
2. **Physics-FiLM modulation**: encodes selected solar–geomagnetic drivers and temporal encodings, then generates bounded channel-wise scale and shift responses for latent VTEC features.
3. **Transformer temporal encoder**: models temporal dependencies over the 24 h input sequence.
4. **Horizon-aware attention pooling**: constructs a horizon-specific temporal context for each of the 24 prediction steps.
5. **Adaptive dual-baseline residual head**: combines the most recent VTEC field and the previous-day same-time prior, followed by residual, gain, and bias corrections.

## Repository structure

```text
HARP-FiLM/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── harp_film.yaml
├── data/
│   └── README.md
├── docs/
│   ├── data_format.md
│   ├── preprocessing.md
│   ├── reproducibility.md
│   ├── open_research_and_references.md
│   └── zenodo_release_checklist.md
├── src/
│   └── harp_film/
│       ├── __init__.py
│       ├── dataset.py
│       ├── model.py
│       ├── train_eval.py
│       └── utils.py
├── scripts/
│   ├── train.py
│   ├── evaluate_hdf5_metrics.py
│   └── train_harp_film_original_reference.py
├── examples/
│   └── demo_model_forward.py
└── outputs/
    └── README.md
```

## Data availability

The original data are publicly available from the following sources:

- **CODE Global Ionosphere Maps (GIMs)**: http://ftp.aiub.unibe.ch/CODE/
- **NASA OMNIWeb / OMNI2 solar-wind and geomagnetic data**: https://omniweb.gsfc.nasa.gov

The processed HDF5 files are not included in this repository because they can be regenerated from the public sources above. The expected HDF5 structure is documented in [`docs/data_format.md`](docs/data_format.md).

## Expected processed files

The main chronological split used in the manuscript is:

| Split | Period | Example processed file |
|---|---:|---|
| Training | 2008–2017 | `data/train_aux_0817_single.h5` |
| Validation | 2018–2019 | `data/val_aux_1819_single.h5` |
| Independent test | 2020 | `data/test_aux_2020_single.h5` |
| Independent test | 2023 | `data/test_aux_2023_single.h5` |

The blocked chronological stress-test setting used for the solar-maximum experiment is described in [`docs/reproducibility.md`](docs/reproducibility.md).

## Selected auxiliary drivers

The manuscript uses five selected solar–geomagnetic drivers plus four temporal encodings:

| Driver group | Variable used in this repository | Physical meaning |
|---|---|---|
| Solar radiative forcing | `f10.7_index` | F10.7 solar radio flux index |
| Auroral activity | `AU-index` | auroral upper index |
| Storm-time geomagnetic disturbance | `DST Index` | Dst index |
| Solar-wind forcing | `Plasma (Flow) speed` | solar-wind proton flow speed |
| Plasma composition | `Na/Np` | alpha-to-proton density ratio |
| Time encoding | `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | local temporal periodicity proxies |

If your processed HDF5 files use different dataset names, update `selected_aux_keys` in [`configs/harp_film.yaml`](configs/harp_film.yaml). To maintain consistency with the manuscript, the model should use the same five selected drivers.

## Installation

```bash
conda create -n harp_film python=3.10 -y
conda activate harp_film
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA environment if needed. For example, follow the official PyTorch installation page for the appropriate CUDA version.

## Training

Edit data paths in `configs/harp_film.yaml`, then run:

```bash
python scripts/train.py --config configs/harp_film.yaml
```

By default, the training script:

- fits the TEC scaler using only training historical frames after `log1p` transformation;
- fits the auxiliary-driver MinMax scaler using only training historical frames;
- trains HARP-FiLM with SmoothL1 loss and AdamW;
- saves the best checkpoint, scalers, auxiliary-key list, training history, and prediction results under `outputs/`.

## Evaluation

After training, the script writes an HDF5 prediction file containing `prediction` and `target` arrays in original TECU scale. Metrics can be recomputed with:

```bash
python scripts/evaluate_hdf5_metrics.py --input outputs/output.h5
```

Reported metrics include RMSE, MAE, R², P90 absolute error, and P95 absolute error.

## Reproducibility notes

The released training pipeline is aligned with the manuscript's main architecture and chronological split. The interpretability figures in the manuscript were generated using diagnostic hooks that extract temporal attention weights, FiLM statistics, and residual-head variables from the trained model. The core modules retain these internal components, and additional diagnostic scripts can be added depending on the exact figure-generation workflow.

See [`docs/reproducibility.md`](docs/reproducibility.md) for detailed experimental settings.

## Citation

If you use this software, please cite the manuscript and the archived software release. After creating a Zenodo release, replace the DOI placeholders in `CITATION.cff` and in the manuscript Open Research section.

```bibtex
@software{chen_2026_harp_film,
  author = {Chen, Jiawen and Yang, C. and Han, L.},
  title = {HARP-FiLM: Global Ionospheric VTEC Forecasting with Physics-Guided Feature Modulation and Dual-Baseline Residuals},
  version = {1.0.0},
  year = {2026},
  publisher = {Zenodo},
  doi = {TO_BE_ASSIGNED_BY_ZENODO},
  url = {https://github.com/research1011/HARP-FiLM}
}
```

## License

This software is released under the MIT License. See [`LICENSE`](LICENSE) for details.
