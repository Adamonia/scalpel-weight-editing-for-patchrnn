# Helper scripts

These scripts check the rebuilt models and run the earlier steps in order.

- `verify_models.py`, `arch_search_tf.py`, `precompute_base.py`: compare each rebuilt model with the stored predictions of the archive.
- `build_raw_cache.py`: builds the text cache of the test samples and the token counts of the training set.
- `*.sh`: start the other scripts one after another.
