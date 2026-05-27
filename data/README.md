# Data directory

Processed data files are not stored in the Git repository because they are large and can be regenerated from public datasets.

Expected processed files for the main experiments:

- `train_aux_0817_single.h5`: training split, 2008–2017
- `val_aux_1819_single.h5`: validation split, 2018–2019
- `test_aux_2020_single.h5`: independent test split, 2020
- `test_aux_2023_single.h5`: independent test split, 2023

Original data sources:

- CODE Global Ionosphere Maps: http://ftp.aiub.unibe.ch/CODE/
- NASA OMNIWeb / OMNI2: https://omniweb.gsfc.nasa.gov

The expected processed HDF5 layout is described in `../docs/data_format.md`.
