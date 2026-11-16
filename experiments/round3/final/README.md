# round3/final

The one-time evaluation of the frozen round 3 recipe on the sealed holdout (3,804 samples).

Files:
- `FROZEN.md`: recipe, hashes and rules, written before any holdout sample was forwarded.
- `code/step1_verify.py`: repeats the diag and guard numbers and compares them with `decisions.json`. No holdout.
- `code/step2a_forward.py`: forwards the holdout once per model and saves the margins.
- `code/step2b_stats.py`: metrics, paired bootstrap, McNemar test, Holm correction. Writes `out/round3_holdout.json`.
- `code/step3_report.py`: builds `out/REPORT.md`.
- `code/fwd.py`, `code/r3lib.py`: helpers.
- `code/scalpel_project/`: a copy of the tool (`scalpel/`, `tests/`) that the scripts import. See its README.
- `code/tmp/S_M*/config.json`: temporary session configs. They are scratch files.

How to run: `step1_verify.py`, `step2a_forward.py`, `step2b_stats.py`, `step3_report.py` in order. Change the paths at the top of the scripts.

Results: `out/REPORT.md`, `out/round3_holdout.json`, `out/summary_all_with_r3.json`.
