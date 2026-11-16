# controls

Control tests for the zero edit of the untrained message-embedding rows. Models: M0, M2, M3, M4, M5, M8, M12, M13.

What it tests:
- Does zeroing the untrained rows beat placebo edits (zero the same number of trained rows, 20 seeds)?
- Do a threshold shift only, a least-squares head, mean imputation or kNN imputation do the same?
- 50 random fit/eval splits of the 7,608 test samples, and a paired bootstrap.
- Holm correction over the 24 published comparisons of rounds 1 and 2.

Files:
- `common.py`: shared helpers (paths, session, edit variants). Paths come from `SCALPEL_*` variables.
- `step1.py MODEL_NO [N_SEEDS]`: margins of all variants for one model.
- `splits.py [MODEL_NO ...]`: 50 splits and bootstrap. Writes `out/splits.json`.
- `holm.py`: Holm correction. Reads `summary_all.json` from rounds 1 and 2. Writes `out/holm.json`.
- `mean_norm.py`: distance of the mean trained row from zero. Writes `out/mean_vec_norm.json`.
- `make_report.py`: builds `out/REPORT.md` from the json files.

How to run (order): `step1.py` for each model, then `splits.py`, `holm.py`, `mean_norm.py`, `make_report.py`.

Results: `out/REPORT.md` (read this first), `out/holm.json`, `out/splits.json`, `out/mean_vec_norm.json`, `out/caveats.txt`.
Per-model files named in the report (`splits_M*.json`, `logits_M*.npz`, `feat_M*.npz`, `meta_M*.json`) are not in the repository.
