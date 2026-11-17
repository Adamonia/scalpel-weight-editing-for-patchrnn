"""CodeBERT (M15), deeper edits than laser_sweep.py. Nothing is trained.

Three families, all on the fine-tuned checkpoint of the archive:
  deep_laser : every one of the 12 layers, matrices q, k, v, attention output, ffn in, ffn out;
               rank-r copy by SVD, rho in {0.05, 0.25, 0.5}.
  prune      : the same 72 matrices, the 20 % smallest weights (by size) set to 0.
  wise       : interpolation with the pre-trained base model (WiSE-FT):
               w = (1 - a) * w_finetuned + a * w_base, for the weights that the base model has,
               groups: all encoder weights, top 4 layers, top 8 layers, word embeddings only; a in {0.1 .. 0.5}.
The base model 'microsoft/codebert-base' is downloaded by the transformers package (about 500 MB, needs internet).

Choice as in laser_report.py: on diag (MCC must not fall), confirmed on guard, holdout is reported.
Output (in --out): codebert_rows.csv, margins_M15.npz, REPORT.md.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from laser_sweep import PARTS, Runner, metrics            # noqa: E402
from laser_report import boot_ci                          # noqa: E402
from scalpel.models import build_model                    # noqa: E402

PRE = 'roberta.encoder.layer.%d.'
MATS = {'q': 'attention.self.query.weight', 'k': 'attention.self.key.weight', 'v': 'attention.self.value.weight',
        'attn_out': 'attention.output.dense.weight', 'ffn_in': 'intermediate.dense.weight',
        'ffn_out': 'output.dense.weight'}
FIELDS = ['family', 'name', 'setting', 'layer'] + ['%s_%s' % (p, m) for p in PARTS for m in ('f1', 'mcc')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', default=os.environ.get('SCALPEL_ARCHIVE', os.path.join(ROOT, '..', 'model-archive')))
    ap.add_argument('--bundle', default=os.path.join(ROOT, 'bundle_codebert'))
    ap.add_argument('--base', default='microsoft/codebert-base')
    ap.add_argument('--out', default=os.path.join(HERE, 'out_codebert'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--limit-rows', type=int, default=0, help='smoke test')
    ap.add_argument('--layers', default='0-11', help='smoke test: for example 10-11')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lo, hi = [int(x) for x in args.layers.split('-')]
    layers = list(range(lo, hi + 1))

    z = np.load(os.path.join(args.bundle, 'bundle.npz'))
    b = {k: z[k] for k in z.files if k in {'y', 'part', 'msg', 'tok_b', 'typ_b', 'tok_a', 'typ_a'}}
    if args.limit_rows:
        b = {k: v[:args.limit_rows] for k, v in b.items()}
    y, part = b['y'].astype(np.int64), b['part']
    idx = {p: np.where(part == v)[0] for p, v in PARTS.items()}
    dev = args.device
    model = build_model(args.archive, 15).to(dev)
    r = Runner(15, model, b, dev, args.batch, dev == 'cuda')
    feats = {s: r.stage(s) for s in model.STAGES if s != 'head'}
    params = dict(model.named_parameters())

    rows_f = open(os.path.join(args.out, 'codebert_rows.csv'), 'w', newline='')
    w = csv.DictWriter(rows_f, fieldnames=FIELDS)
    w.writeheader()
    margins, t0 = {}, time.time()

    def evaluate(family, name, setting, layer):
        f2 = dict(feats)
        f2['msg'] = r.stage('msg')
        mg = r.head(f2)
        margins['%s|%s|%s' % (family, name, setting)] = mg.astype(np.float16)
        d = {'family': family, 'name': name, 'setting': setting, 'layer': layer}
        for p in PARTS:
            f1, mcc = metrics(y, mg, idx[p])
            d[p + '_f1'], d[p + '_mcc'] = round(f1, 4), round(mcc, 5)
        w.writerow(d)
        rows_f.flush()
        return mg

    evaluate('original', 'original', '-', -1)
    # ---- deep LASER and pruning
    for L in layers:
        for short, suffix in MATS.items():
            p = params[PRE % L + suffix]
            w0 = p.data.clone()
            U, S, Vh = torch.linalg.svd(w0.float(), full_matrices=False)
            for rho in (0.05, 0.25, 0.5):
                k = max(1, int(round(rho * S.numel())))
                p.data = ((U[:, :k] * S[:k]) @ Vh[:k]).to(w0.dtype)
                evaluate('deep_laser', '%s_L%d' % (short, L), rho, L)
            thr = torch.quantile(w0.abs().float().flatten()[::7], 0.20)
            p.data = torch.where(w0.abs() < thr, torch.zeros_like(w0), w0)
            evaluate('prune', '%s_L%d' % (short, L), 0.2, L)
            p.data = w0
        print('layer %d done, %.0f s' % (L, time.time() - t0), flush=True)
    # ---- WiSE-FT interpolation
    from transformers import RobertaModel
    base = RobertaModel.from_pretrained(args.base)
    bsd = {'roberta.' + k: v for k, v in base.state_dict().items()}
    groups = {'all': lambda n: True,
              'top4': lambda n: any((PRE % i) in n for i in range(8, 12)),
              'top8': lambda n: any((PRE % i) in n for i in range(4, 12)),
              'embeddings': lambda n: 'embeddings.word_embeddings' in n}
    saved = {n: p.data.clone() for n, p in params.items() if n in bsd and p.shape == bsd[n].shape}
    for gname, sel in groups.items():
        for a in (0.1, 0.2, 0.3, 0.4, 0.5):
            for n, w0 in saved.items():
                params[n].data = w0 if not sel(n) else (1 - a) * w0 + a * bsd[n].to(w0.device, w0.dtype)
            evaluate('wise', gname, a, -1)
    for n, w0 in saved.items():
        params[n].data = w0
    np.savez_compressed(os.path.join(args.out, 'margins_M15.npz'), **margins)
    rows_f.close()

    # ---- choice and report
    rows = list(csv.DictReader(open(os.path.join(args.out, 'codebert_rows.csv'))))
    for x in rows:
        for p in PARTS:
            x[p + '_f1'], x[p + '_mcc'] = float(x[p + '_f1']), float(x[p + '_mcc'])
    o = rows[0]

    def d(x, p):
        return x[p + '_f1'] - o[p + '_f1'], 100 * (x[p + '_mcc'] - o[p + '_mcc'])
    L = ['# CodeBERT (M15), deeper edits', '',
         'Original: F1 diag %.2f, guard %.2f, holdout %.2f; MCC holdout %.3f. Changes in F1 points, MCC x 100.' %
         (o['diag_f1'], o['guard_f1'], o['holdout_f1'], o['holdout_mcc']), '',
         '| Family | Best on diag | Diag | Guard | Holdout | Holdout dF1 [95 %] | Used |', '|---|---|---|---|---|---|---|']
    hold = np.where(part == 2)[0]
    for fam in ('deep_laser', 'prune', 'wise'):
        c = [x for x in rows if x['family'] == fam and d(x, 'diag')[1] >= 0 and d(x, 'diag')[0] > 0]
        if not c:
            L.append('| %s | none | | | | | |' % fam)
            continue
        x = max(c, key=lambda x: d(x, 'diag')[0])
        ok = d(x, 'guard')[0] >= 0 and d(x, 'guard')[1] >= 0
        key = '%s|%s|%s' % (fam, x['name'], x['setting'])
        ci = boot_ci(y, margins['original|original|-'].astype(np.float64), margins[key].astype(np.float64), hold)
        L.append('| %s | `%s` %s | %+.2f / %+.2f | %+.2f / %+.2f | %+.2f / %+.2f | %+.2f [%+.2f, %+.2f] | %s |' % (
            fam, x['name'], x['setting'], *d(x, 'diag'), *d(x, 'guard'), *d(x, 'holdout'), d(x, 'holdout')[0], *ci,
            'yes' if ok else 'no (guard falls)'))
    L += ['', '%d edited models were tried. Gains on diag are expected by chance at this number; only guard and holdout count.' % (len(rows) - 1)]
    open(os.path.join(args.out, 'REPORT.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
