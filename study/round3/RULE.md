# Round 3 rule (written before any guard number was looked at)

Data: diag (part 0) explores and chooses, guard (part 1) confirms. Holdout (part 2) is never used.
Margin = logit[1] - logit[0]. Prediction = margin > 0. Change = edited minus reference, in F1 points.
Recenter (rc) = label-free shift of all margins of a model so that the mean margin on DIAG equals the mean
margin of the original on DIAG (same shift is used on guard, as the engine's edit_recenter did in Round 2).

Candidates (22): edit in {E2, E3 k=1,3,10,30,100, E4 k=1,3,10,30,100} x rc in {yes, no}.
Score of a candidate = mean over the six word-level Transformers (M3, M4, M5, M8, M12, M13) of the
diag F1 change against E1 with the same rc flag.
1. Best score wins. Candidates within 0.05 F1 points of the best score are tied; the tie goes to the simpler
   edit (E2, then E4, then E3), then to rc = no, then to the smaller k.
2. Noise gate: the winner must have score > 0 and the lower end of the paired bootstrap 95 % interval
   (2,000 resamples of the diag samples, the same resample for all six models, mean over the six models)
   must be > 0. If not, the recipe is E1 (rc chosen by the larger mean diag F1 change against E0, tie -> no).
3. Guard confirms or not: reported plainly, the recipe is NOT changed after looking at guard.
Per-model decision (all 12 models): the recipe is applied to a model only if guard F1 of the recipe is not
below guard F1 of the original (E0). Otherwise the model keeps its original weights.
