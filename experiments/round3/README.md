# round3

Round 3: one fixed edit for all models, without agent choices. Every message-embedding row is multiplied by c/(c+3), where c is the training count of the token (untrained rows become 0). The pad row is not changed.

Protocol:
1. `RULE.md` fixes the rule to choose the recipe. It was written before any guard number was read.
2. The recipe is chosen on diag only (22 candidates), checked on guard, then tested once on the sealed holdout (`final/`).
3. A model gets the recipe only if its guard F1 is not below the original.

Files:
- `r3lib.py`, `analysis.py`: helpers and metrics. `r3lib.py` needs a `scalpel_project/` folder next to it (a copy of this repository's root; see `final/code/scalpel_project/`). Change the paths at the top.
- `compute_margins.py MODEL_NO`: margins of all candidates on diag and guard. The holdout is never forwarded.
- `select_recipe.py`: chooses the recipe on diag. Writes `out/choice_diag.json`.
- `finalize.py`: metrics on diag and guard, `recipe.json`, `decisions.json`.
- `make_report.py`: builds `out/REPORT.md`.
- `apply_round3.py`, `common.py`, `model.py`: copies of the retraining kit files for the six archive architectures. The kit in `../retrain_kit/` is the newer version. Use that one.
- `final/`: the final holdout run. See `final/README.md`.

How to run (order): `compute_margins.py` for each of the 12 models, `select_recipe.py`, `finalize.py`, `make_report.py`. Then the steps in `final/`.

Results: `out/REPORT.md`, `out/recipe.json`, `out/decisions.json`, `final/out/REPORT.md`. Some files named in `out/REPORT.md` (`metrics.json`, `summary.json`, `choice_diag.json`, `margins_M*.npz`, `meta_M*.json`) are not in the repository.
