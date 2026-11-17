# UNK rule (written before any UNK result exists)

Question: does normal training with an UNK token give the same or more than the zero edit, with no edit?

Setup. Same data, split, seeds, epochs, validation selection and hyperparameters as the first 50 runs
(10 architectures x 5 seeds). One change: every message token with training count c <= 1 is mapped to one
new UNK id (train, validation and test). The count c of a token is the number of fit rows (training rows
without the validation rows) that contain the token in the model window. Pad id and code input are not changed.
Runs pair with the first 50 runs by (architecture, seed). The first 50 runs are in scalpel_runs_v2. The UNK runs are in scalpel_runs_unk.

Numbers per pair (F1 on the test part, in percent, prediction = margin > 0, no shift of margins):
- base: the first run, no edit.
- zero edit: the same run, message rows of tokens with c = 0 set to 0 (E1 raw in round3_results.csv).
- round 3: the same run with the round 3 recipe of recipe.json (E3, k = 3, no recenter).
- UNK: the UNK run, no edit.

PRIMARY TEST (only one):
  d = F1(UNK) - F1(base + zero edit), over the 50 pairs.
  Report the mean of d, the 95 % paired bootstrap interval (10,000 resamples of the pairs,
  numpy default_rng seed 12345, percentile 2.5 and 97.5) and the two-sided exact sign test
  (pairs with d = 0 are dropped).
  Reading: UNK is "better" if the whole interval is above 0, "worse" if the whole interval is below 0.
  If the interval holds 0, the data show no difference.

SECONDARY (no correction, reported as they are):
  1. F1(UNK) - F1(base).
  2. F1(UNK) - F1(base + round 3).
  3. The slope of d (and of the two differences above) over log2(heads), as in analyse_seeds.py
     (trend(): run-level fit and the architecture-level permutation test inside each layer stratum).

Do not change this rule after the first UNK result is seen. If a run fails or is missing, say so
and do not replace it.
