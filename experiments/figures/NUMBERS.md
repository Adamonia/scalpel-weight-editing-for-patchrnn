# Numbers shown in the figures

## Fig 1: precision-recall on holdout (n=3804), threshold 0
- M3 orig: precision 0.7002, recall 0.7553 (F1 72.67)
- M3 untrained_zero: precision 0.7242, recall 0.7618 (F1 74.25)
- M4 orig: precision 0.6954, recall 0.7382 (F1 71.62)
- M4 untrained_zero: precision 0.7400, recall 0.7172 (F1 72.84)

## Fig 2: F1 change in pp, full set (n=7608). Edited = untrained_zero; placebo = 20 freqmatch seeds
- M0: edited +0.06 pp; placebo mean -0.13, min -0.35, max +0.13
- M5: edited +0.25 pp; placebo mean -0.24, min -0.38, max -0.13
- M12: edited +0.27 pp; placebo mean -0.10, min -0.24, max +0.12
- M2: edited +0.41 pp; placebo mean +0.20, min -0.13, max +0.38
- M13: edited +0.58 pp; placebo mean -0.26, min -0.43, max -0.09
- M8: edited +0.64 pp; placebo mean -0.32, min -0.52, max -0.08
- M4: edited +1.03 pp; placebo mean +0.06, min -0.16, max +0.27
- M3: edited +1.14 pp; placebo mean +0.12, min -0.07, max +0.34

## Fig 3: M3 holdout, n=3804
- Wrong to right: 72
- Right to wrong: 23
- Unchanged right: 3080
- Unchanged wrong: 629

## Fig 4: example, M3 holdout sample
- global sample index 1636 (position 812 inside holdout), true label 0
- token ids: [724, 76411, 314, 357, 34529, 2318]
- tokens: ['improv', 'print_expr', 'littl', 'still', 'bright', 'though']
- untrained flags: [False, True, False, False, False, False]
- margin before 1.204, after -4.391 (change -5.595); prediction 1 -> 0

## Fig 5: M3 message embedding rows (first 24 of 128 dims); colour range +-2.9549
- trained rows (unchanged): 2_spdb_struct (id 28898), ldebugfs_fid_write_common (id 22789), check_chain_key (id 25506), procglobalshmems (id 67932), pmuv (id 58371), send_v (id 48591)
- untrained rows (set to zero): if0a3e4f3b9598f586fa460d0d07032d22e203122 (id 53822), pgdat_reclaimable_pag (id 79542), mdiobus_alloc_s (id 15022), mutex-protect (id 70626), <c1009707> (id 647), 31aa8 (id 39910)

## Not done / caveats
- Fig 6 is a schematic, no numbers. Accept/refuse logic is drawn as described in the task, not read from code.
- Fig 5 rows are a random draw (seed 1) of trained and untrained tokens; dims are the first 24 only.
- Fig 4 token text decoded with the M3 message vocabulary; window is first 199 tokens, padding removed.
