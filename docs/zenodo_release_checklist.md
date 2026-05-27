# Zenodo release checklist

Use this checklist before resubmitting the manuscript to AGU/JGR.

## GitHub repository

- [ ] Repository is not empty.
- [ ] `README.md` explains the model, data sources, data format, training, evaluation, and citation.
- [ ] `requirements.txt` is included.
- [ ] `LICENSE` is included.
- [ ] `CITATION.cff` is included.
- [ ] Data files are not committed if they are large or regenerated from public data.
- [ ] `docs/data_format.md` clearly describes the expected HDF5 structure.
- [ ] The repository name and all documentation use `HARP-FiLM`, not `HAPR-FiLM`.

## GitHub release

- [ ] Commit all final files.
- [ ] Create a release named `v1.0.0`.
- [ ] Add a short release note, e.g., `Initial public release accompanying the JGR: Space Physics manuscript.`

## Zenodo

- [ ] Connect the GitHub repository to Zenodo.
- [ ] Enable archiving for `research1011/HARP-FiLM`.
- [ ] Archive the `v1.0.0` release.
- [ ] Copy the generated Zenodo DOI.
- [ ] Update `CITATION.cff` with the DOI if desired.
- [ ] Update the manuscript Open Research section with the Zenodo DOI.
- [ ] Add the software Reference List entry with `[Software]`.

## Manuscript resubmission

- [ ] Upload the revised manuscript.
- [ ] Upload each figure separately.
- [ ] Remove figure titles, captions, and line numbers from figure files.
- [ ] Keep subfigures of the same figure number combined into one file.
- [ ] Keep supporting figures inside the Supporting Information file.
