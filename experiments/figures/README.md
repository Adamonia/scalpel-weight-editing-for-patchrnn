# figures

Scripts and images for the figures of the article. `NUMBERS.md` lists the numbers shown in figures 1 to 6.

| Script | Makes | Input |
|---|---|---|
| `make_figs.py` | fig1 to fig6 (`fig1_pr_curve`, `fig2_placebo`, `fig3_flow`, `fig4_example`, `fig5_embedding`, `fig6_loop`) | Files from `controls/out` and the local bundle. |
| `make_forest.py` | `fig0_gains` (rounds 1 and 2) | `controls/out/holm.json`, `results/summary_all.json` |
| `make_forest3.py` | `fig0_gains3` (rounds 1, 2 and 3) | the same, and `round3/final/out/round3_holdout.json` |
| `make_rounds_heatmap.py` | `fig10_rounds_embedding` | round 3 files |
| `make_seed_fig.py` | `fig7_seeds` | `results.csv` in the current folder |
| `make_fig50.py` | `fig8_grid50`, `fig9_mechanism` | `results.csv`, `round3_results.csv`, `mechanism.csv` in the current folder |

How to run: `python make_figs.py` and the other scripts, one by one. Needs matplotlib and numpy. The scripts for fig7 to fig9 read the csv files from the current folder. Run them in `../retrain_kit/out/` or copy the csv files here. The figure scripts read the result files in `../controls/out/`, `../round3/` and `../../results/` and write the images to this folder.

Results: the png files in this folder.
