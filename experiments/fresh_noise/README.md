# fresh_noise

Tests if the gain of the zero edit comes from the architecture or from one random start. The untrained rows are replaced by new random draws (30 draws, N(0,1)). The script also measures the attention to untrained tokens and the change per head. Models: M3, M4, M5, M8, M12, M13.

Files:
- `common.py`: shared helpers. Paths come from `SCALPEL_*` variables.
- `step1.py MODEL_NO`: reproduction check, fresh-noise test, clean-margin density. Writes `out/t1t2_M{n}.json`.
- `step3.py MODEL_NO`: attention to untrained tokens in layer 0 and change per head. Writes `out/t3t4_M{n}.json`.
- `step4.py`: builds `out/REPORT.md` from the json files.

How to run: `step1.py` and `step3.py` for each of the six models, then `step4.py`.

Results: `out/REPORT.md`, `out/summary.json`, `out/t1t2_M*.json`, `out/t3t4_M*.json`.
`step1.py` also writes `out/margins_fresh_M{n}.npz`. These files are not in the repository.
