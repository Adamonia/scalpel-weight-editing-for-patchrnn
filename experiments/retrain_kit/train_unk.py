"""Train all 10 architectures x 5 seeds with UNK (K = 1). Same as train_seeds.py, only --unk 1 and another folder.

Example: python train_unk.py --data dataset.npz
Other options are the same as in train_seeds.py. Finished runs are skipped.
"""
import os
import sys
from pathlib import Path

import train_seeds

BASE_FOLDER = 'scalpel_runs_v2'


def main():
    argv = sys.argv[1:]
    if '--unk' not in argv:
        argv += ['--unk', '1']
    if '--out' not in argv:
        argv += ['--out', str(Path(os.path.expanduser('~')) / 'scalpel_runs_unk')]
    out = Path(argv[argv.index('--out') + 1])
    if out.name == BASE_FOLDER:
        raise SystemExit('Do not write UNK runs into %s.' % BASE_FOLDER)
    sys.argv = [sys.argv[0]] + argv
    train_seeds.main()


if __name__ == '__main__':
    main()
