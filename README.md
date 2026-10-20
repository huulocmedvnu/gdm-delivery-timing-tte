# Delivery at 39 weeks compared with expectant management in gestational diabetes by body mass index

Analysis code for a target trial emulation of delivery in week 39 compared with expectant management in singleton,
nonanomalous, cephalic pregnancies with gestational diabetes ongoing at 39 0/7 weeks, United States, 2019 to 2024, by
prepregnancy body mass index, with secondary trials starting at 38 and 40 weeks. The data are the public-use natality,
fetal death and linked birth/infant death period files of the National Center for Health Statistics (NCHS). No
individual-level data are included in this repository.

## Contents

| Path | Purpose |
|---|---|
| `rust/` | `vitals`: multithreaded fixed-width parser of the NCHS files. Applies the eligibility criteria and writes counts by covariate pattern (`gdm`), the study flow (`gdmflow`) and birthweight distributions (`bwq`). Field positions follow the NCHS user guides. |
| `rust_ccw/` | `ccw`: clone-censor-weight estimation. Logistic models for the probabilities of induction and birth in the trial week and of continuation to the next week within each BMI class, weights truncated at the 99th percentile, normalized risks, and a Bayesian bootstrap that refits both models in every draw. |
| `scripts/download*.sh` | Download the NCHS files and user guides. |
| `scripts/bw_thresholds.py` | Birthweight 10th percentile by week and sex. |
| `scripts/gdm_ccw.py` | Trials at 38, 39 and 40 weeks, covariate balance and sensitivity analyses. Calls `rust_ccw`. |
| `scripts/psm.py` | Propensity score matched conventional comparison (post hoc). |
| `scripts/manuscript_numbers.py`, `scripts/figures.py` | Tables, reported numbers and figures. |
| `scripts/validate_rust.py` | Check of `rust_ccw` against statsmodels on identical data. |
| `scripts/util.py` | Shared helpers (design matrix, E-value). |
| `scripts/run_all.sh` | Whole pipeline. |

## Running

Requirements: Rust (tested with 1.94), Python 3.13 with `requirements.txt`, `unzip`, and 7-Zip (`7zz`; the 2015
natality and 2016 fetal death archives use PPMd compression). About 6 GB of downloads.

```bash
PY=python NCPU=12 SEVENZIP=7zz bash scripts/run_all.sh
```

Results are written to `outputs/` (JSON and TSV) and `figures/`. The clone-censor-weight step for all three trials,
with 1,000 refitted bootstrap draws each, takes about 100 seconds with 48 threads. Random seeds are fixed in the
scripts.

## Validation

`rust_ccw` was checked against statsmodels generalized linear models on the same covariate-pattern counts
(`scripts/validate_rust.py`): treatment model coefficients agreed to within 1e-12.

## License

MIT. See `LICENSE`.
