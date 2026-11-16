# tools

Helper scripts from the setup of the study. They rebuild the original models, check them, and start earlier steps in order. They have no results of their own.

Python scripts:
- `arch_search_tf.py MODEL_NO [N]`: searches the forward settings of the TwinTransformer that reproduce `predicted.txt` of the archive (default N = 96 samples).
- `verify_models.py N [MODEL_NO ...]`: checks each rebuilt model against `predicted.txt` on N samples. Default: all models in the registry.
- `precompute_base.py MODEL_NO ...`: computes the base stage outputs of archive models and compares them with `predicted.txt`.
- `build_raw_cache.py`: builds the text cache of the test samples and the token counts of the training set.

Shell scripts (they start the scripts above and `study/` scripts in order):
- `init_new.sh`: starts sessions for M6 and M7. `init_m15.sh`: builds the bundle and starts the session for M15.
- `precompute_all.sh`: `precompute_base.py` for M3, M4, M13, M5, M8, M12.
- `battery_all.sh`, `battery_new.sh`: the diagnostic battery (`study/battery.py`) for the archive models and for M14, M6, M7.
- `finalize_all.sh`: finalizes the sessions of the eight base models and builds the tables (`study/aggregate.py`).

The paths in the scripts are absolute paths of the study machine. Change them before a run.
