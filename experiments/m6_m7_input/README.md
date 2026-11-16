# m6_m7_input

Tests if M6 (BERT word pieces) and M7 (RoBERTa byte-pair ids) can be repaired without training by a change of the input format. Variants: training format, single-space tokenization, each abstract token alone, copy of the boundary twin into untrained rows, mean fill.

Files:
- `common.py`: shared code (tokenization variants, model run, metrics, bootstrap). Change the paths at the top.
- `facts_probe.py`: shows how the tokenizer splits the training-time text (probe only).
- `step0_check.py`: checks that the encoding equals the bundle and the model gives the archive predictions.
- `step1_facts.py`: facts about formats, untrained rows and twins. Writes `out/facts.json`.
- `step2_eval.py [6] [7]`: logits for each variant.
- `step3_analyze.py`: metrics and paired bootstrap. Writes `out/results.json` and `out/results.txt`.
- `step4_branch.py`: which input branch causes the loss. Writes `out/branch_ablation.json`.
- `step5_report.py`: builds `out/REPORT.md` from `report_template.md` and the json files. Run it in this folder.

How to run: `step0_check.py` to `step5_report.py` in order.

Results: `out/REPORT.md` (verdict in the first section: the input defect cannot be repaired without training). The logit cache is not in the repository.
