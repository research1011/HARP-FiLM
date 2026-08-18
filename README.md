# HARP-FiLM: Global Ionospheric VTEC Forecasting

This repository provides the PyTorch implementation of **HARP-FiLM** (*Horizon-Aware Adaptive Residual Prediction with Feature-wise Linear Modulation*) for 24 h global ionospheric vertical total electron content (VTEC) forecasting.

The repository is prepared for the manuscript:

> **HARP-FiLM: Global Ionospheric VTEC Forecasting With Solar–Geomagnetic Driver Conditioning and Dual-Baseline Residuals**

HARP-FiLM combines multi-scale spatial encoding, solar–geomagnetic driver-conditioned feature-wise modulation, Transformer-based temporal modeling, horizon-aware temporal attention pooling, and adaptive dual-baseline residual forecasting.

## Model overview

The implementation corresponds to the method described in the manuscript:

1. **Multi-scale CNN spatial encoder**: extracts global VTEC morphology using 3 × 3, 5 × 5, and 7 × 7 convolutional branches.
2. **Solar–Geomagnetic FiLM conditioning**: encodes selected solar–geomagnetic drivers and temporal encodings and generates bounded channel-wise scale responses together with additive feature updates for latent VTEC representations.
3. **Transformer temporal encoder**: models temporal dependencies over the 24 h historical input sequence.
4. **Horizon-aware attention pooling**: constructs a horizon-specific temporal context for each of the 24 prediction steps.
5. **Adaptive dual-baseline residual head**: combines the most recent VTEC field and the previous-day same-time prior using a sample- and horizon-dependent fusion coefficient, followed by residual, gain, and bias corrections.

The revised manuscript uses IMF Bz as an additional auxiliary driver. This addition changes only the auxiliary-input configuration; the HARP-FiLM architecture and training implementation remain unchanged.

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

* **CODE Global Ionosphere Maps (GIMs)**: http://ftp.aiub.unibe.ch/CODE/
* **NASA OMNIWeb / OMNI2 solar-wind and geomagnetic data**: https://omniweb.gsfc.nasa.gov
* **ISEE global GNSS-derived TEC products**: https://stdb2.isee.nagoya-u.ac.jp/GPS/shinbori/AGRID2/nc/

The processed HDF5 files are not included in this repository because they can be regenerated from the publicly available source data. The expected HDF5 structure is documented in [`docs/data_format.md`](docs/data_format.md).

The ISEE GNSS-derived TEC product is used only for post-training external cross-product evaluation and is not involved in model training, validation, model selection, fine-tuning, or calibration.

## Expected processed files

The main chronological split used in the manuscript is:

| Split            |    Period | Example processed file          |
| ---------------- | --------: | ------------------------------- |
| Training         | 2008–2017 | `data/train_aux_0817_single.h5` |
| Validation       | 2018–2019 | `data/val_aux_1819_single.h5`   |
| Independent test |      2020 | `data/test_aux_2020_single.h5`  |
| Independent test |      2023 | `data/test_aux_2023_single.h5`  |

The blocked chronological stress-test setting used for the solar-activity evaluation is described in [`docs/reproducibility.md`](docs/reproducibility.md).

## Selected auxiliary drivers

The revised manuscript uses **six solar–geomagnetic drivers plus four temporal encodings**:

| Driver group                       | Variable used in this repository             | Physical meaning                                     |
| ---------------------------------- | -------------------------------------------- | ---------------------------------------------------- |
| Solar radiative forcing            | `f10.7_index`                                | F10.7 solar radio flux index                         |
| Auroral activity                   | `AU-index`                                   | auroral upper index                                  |
| Storm-time geomagnetic disturbance | `DST Index`                                  | Dst index                                            |
| Solar-wind forcing                 | `Plasma (Flow) speed`                        | solar-wind proton flow speed                         |
| Plasma composition                 | `Na/Np`                                      | alpha-to-proton density ratio                        |
| Upstream IMF forcing               | `Bz_GSM`                                     | GSM z-component of the interplanetary magnetic field |
| Time encoding                      | `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos` | temporal periodicity encodings                       |

The first five solar–geomagnetic variables were retained through the feature-screening procedure described in the manuscript. IMF Bz in GSM coordinates was additionally included as an upstream descriptor of solar-wind–magnetosphere coupling.

If your processed HDF5 files use different dataset names, update `selected_aux_keys` in [`configs/harp_film.yaml`](configs/harp_film.yaml). To maintain consistency with the revised manuscript, the model should use the same six solar–geomagnetic drivers together with the four temporal encodings.

**Important:** `Bz_GSM` above is an example dataset key. If the processed HDF5 files use a different name for IMF Bz in GSM coordinates, replace it with the exact key used in those files and in `configs/harp_film.yaml`.

The addition of IMF Bz changes only the auxiliary-input configuration. No modification to the core HARP-FiLM architecture is required as long as the auxiliary input dimension is configured consistently with the selected driver set.

## Installation

```bash
conda create -n harp_film python=3.10 -y
conda activate harp_film
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA environment if needed. For example, follow the official PyTorch installation instructions for the appropriate CUDA version.

## Training

Edit the data paths and auxiliary-variable configuration in `configs/harp_film.yaml`, then run:

```bash
python scripts/train.py --config configs/harp_film.yaml
```

By default, the training script:

* fits the VTEC scaler using only training-period historical frames after `log1p` transformation;
* fits the auxiliary-driver MinMax scaler using only training-period historical auxiliary inputs;
* retains test-period auxiliary values outside the fitted MinMax range rather than clipping them;
* trains HARP-FiLM with SmoothL1 loss and AdamW;
* saves the best checkpoint, scalers, auxiliary-key list, training history, and prediction results under `outputs/`.

Only auxiliary-driver observations within the 24 h historical input window are supplied to the forecasting model. Future solar-wind, IMF, or geomagnetic observations after forecast initialization are not used as model inputs.

## Evaluation

After training, the script writes an HDF5 prediction file containing `prediction` and `target` arrays in the original TECU scale. Metrics can be recomputed with:

```bash
python scripts/evaluate_hdf5_metrics.py --input outputs/output.h5
```

Reported CODE-referenced forecasting metrics include:

* RMSE;
* MAE;
* R²;
* P90 absolute error;
* P95 absolute error.

The manuscript additionally reports bias and Pearson correlation for the external ISEE-referenced cross-product evaluation.

## Reproducibility notes

The released training pipeline is aligned with the manuscript's HARP-FiLM architecture and chronological experimental design.

For the main experiment:

* training data: 2008–2017;
* validation data: 2018–2019;
* held-out test years: 2020 and 2023.

Additional blocked chronological experiments retrain the forecasting models using only pre-test-period data for evaluation in 2014 and 2017. Details of the blocked experiments are provided in [`docs/reproducibility.md`](docs/reproducibility.md).

The manuscript also includes temporal-interpolation sensitivity experiments. In the standard hourly experiment, the earlier native 2 h CODE GIM products are temporally harmonized to an hourly sequence. Complementary sensitivity experiments evaluate the effect of excluding interpolated target epochs from optimization and of using exclusively native 2 h CODE sequences.

The interpretability and internal-response figures in the manuscript were generated using diagnostic hooks that extract temporal attention weights, FiLM modulation statistics, spatial sensitivity information, and adaptive dual-baseline residual-head variables from the trained model. The core modules retain these internal components, and additional diagnostic scripts can be used depending on the exact figure-generation workflow.

The 2014 distribution-shift diagnostic evaluates the internal FiLM response under F10.7 values that exceed the MinMax range observed during the corresponding training period. These analyses are intended as diagnostics of the trained model response and should not be interpreted as evidence of unrestricted extrapolation beyond the tested solar-activity range.

The rapid southward-IMF analysis reported in the manuscript is a retrospective short-lead robustness diagnostic. Future IMF Bz values after forecast initialization are used only to identify evaluation intervals and are not supplied to the forecasting model.

See [`docs/reproducibility.md`](docs/reproducibility.md) for detailed experimental settings.

## External cross-product evaluation

In addition to CODE-referenced forecasting evaluation, the manuscript evaluates frozen model forecasts against the independently processed ISEE global GNSS-derived TEC product for 2014, 2017, 2020, and 2023.

The ISEE data are used exclusively for post-training evaluation. They are not used for:

* model training;
* validation;
* model selection;
* fine-tuning;
* calibration.

Because HARP-FiLM forecasts are generated on the 2.5° × 5° CODE grid, the higher-resolution ISEE observations are aggregated to the corresponding forecasting grid cells before evaluation. The resulting comparison should therefore be interpreted as an external cross-product evaluation at the forecasting-grid scale rather than receiver-level validation.

## Citation

If you use this software, please cite the manuscript and the archived software release.

Archived software release:

**Chen, J., Yang, C., & Han, L. (2026). HARP-FiLM: Global Ionospheric VTEC Forecasting With Solar–Geomagnetic Driver Conditioning and Dual-Baseline Residuals (Version 1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20406934**

```bibtex
@software{chen_2026_harp_film,
  author    = {Chen, Jiawen and Yang, C. and Han, L.},
  title     = {HARP-FiLM: Global Ionospheric VTEC Forecasting With Solar--Geomagnetic Driver Conditioning and Dual-Baseline Residuals},
  version   = {1.0.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20406934},
  url       = {https://github.com/research1011/HARP-FiLM}
}
```

The corresponding repository is available at:

`https://github.com/research1011/HARP-FiLM`

The software citation information should also be kept synchronized with `CITATION.cff`.

## License

This software is released under the MIT License. See [`LICENSE`](LICENSE) for details.
