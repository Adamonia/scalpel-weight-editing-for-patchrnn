# stage_trace

Looks for the stage where the noise-to-signal ratio (NSR) of M3, M4 and M13 becomes larger than that of M5, M8 and M12. The script runs the message branch by hand, stage by stage, with 10 fresh N(0,1) draws in the untrained rows.

Files:
- `common.py`: shared helpers. Paths come from `SCALPEL_*` variables.
- `step1.py MODEL_NO`: stage trace, attention shape, clean signal, sample composition. Writes `out/trace_M{n}.json` and `tmp/arrays_M{n}.npz`.
- `step2.py MODEL_NO`: readout gain at the pooled message vector. Writes `out/readout_M{n}.json`.
- `step3.py`: weight norms of the six models. Writes `out/weights.json`.
- `step5.py`: bootstrap of the NSR over test samples. Writes `out/bootstrap.json`.
- `step4.py`: builds `out/REPORT.md` from the json files and `verdict.md`.
- `verdict.md`: conclusion text, written by hand after the tables. `step4.py` reads it.

How to run: `step1.py` and `step2.py` for each model (M3, M4, M5, M8, M12, M13), then `step3.py`, `step5.py`, `step4.py`. Create a `tmp/` folder first.

Results: `out/REPORT.md` and `verdict.md`. The numbers are in `out/*.json`.
The arrays in `tmp/` are not in the repository.
