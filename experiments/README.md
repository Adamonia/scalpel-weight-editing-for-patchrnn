# Experiments

All studies after the three rounds (the rounds are in `../study/`). Each folder has its code, a short `README.md` (what it tests, how to run, where the results are) and results in `out/`.

Before you run a script:
- The paths come from the variables `SCALPEL_WORK`, `SCALPEL_ARCHIVE`, `SCALPEL_BUNDLE` (and `SCALPEL_PUB_TEMP`, `SCALPEL_RUNS`). The defaults are folders of the repository (see the root `README.md`). The shell scripts use `PYTHON` for the Python command.
- Model weights, cached logits, bundles and the dataset are not in the repository. Some files named in the reports (for example `logits_M*.npz` or `margins_M*.npz`) are not in `out/`.
- The reports in `out/` are the result files to read. The json and csv files next to them hold the numbers.

| Folder | Question | Result |
|---|---|---|
| `controls` | Is the gain of the zero edit real? Placebo rows, threshold only, least-squares head, imputation, 50 splits, Holm correction. | Real for M3 and M4 after correction (round 2 edit). M8 and M13 beat the placebo only. |
| `m6_m7_input` | Can M6 and M7 be repaired by a change of the input format without training? | No. Normal tokenization makes both models worse. |
| `fresh_noise` | Is the gain a property of the architecture or of one random start? 30 fresh random draws for the untrained rows. | Part of the pattern is luck of the start. The link to the head count is weak with one model per architecture. `retrain_kit` tests this with 50 models. |
| `stage_trace` | Where does the larger noise-to-signal of M3, M4 and M13 come from? | It appears before the classifier, in the pooled message vector. The cause is not found with one model per architecture. |
| `retrain_kit` | Train a grid of 10 architectures (1, 2, 4, 8, 16 heads; 1 and 2 layers) with 5 seeds each. Compare architecture and seed effects, test the mechanism, apply the round 3 recipe. | The zero edit raises F1 in all 50 models (mean +0.93). Each doubling of the heads lowers the gain by 0.185 points (exact permutation p = 0.0004). Noise travels through the attention weights. Round 3 adds +0.33 points over the zero edit. |
| `figures` | Scripts and images for the article. | Figures 0 to 10 and `NUMBERS.md`. |
| `tools` | Helper scripts that rebuild and check the models and run the earlier steps in order. | No results of its own. |
