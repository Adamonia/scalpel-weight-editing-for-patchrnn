# Experiments

Follow-up studies for the weight edits. Each folder has the code and a report in `out/`.
The scripts use absolute paths of the study machine. Change the paths at the top of each `common.py` before a run.
Model weights, cached logits and the dataset are not in the repository.

| Folder | Question | Result |
|---|---|---|
| `controls` | Is the gain of the zero edit real? Placebo rows, threshold only, least-squares head, imputation, 50 splits, Holm correction. | Real for M3 and M4 after correction. M8 and M13 beat the placebo only. |
| `m6_m7_input` | Can M6 and M7 be repaired by a change of the input format without training? | No. Normal tokenization makes both models worse. |
| `fresh_noise` | Is the gain a property of the architecture or of one random start? 30 fresh random draws for the untrained rows. | Part of the pattern is luck of the start. The link to the head count is weak. |
| `stage_trace` | Where does the larger noise-to-signal of M3, M4 and M13 come from? | It appears before the classifier, in the pooled message vector. The cause is not found. |
| `figures` | Scripts and images for the article. | |
| `retrain_kit` | Train a grid of 10 architectures (1, 2, 4, 8, 16 heads; 1 and 2 layers) with 5 seeds each. Compare architecture and seed effects, test the mechanism, apply the Round 3 recipe. | The zero edit raises F1 in all 50 models. Each doubling of the heads lowers the gain by 0.18 points (exact permutation p < 0.001). Noise travels through the attention weights; many heads average it away. Round 3 adds +0.33 points over the zero edit. Results in `out/`. The dataset file is not included. |
