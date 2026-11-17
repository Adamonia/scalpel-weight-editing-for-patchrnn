"""LASER sweep: replace one weight matrix by a low-rank copy and measure F1 and MCC.

For every model of the archive and every weight matrix (2-D or more), the matrix W is replaced by its
best rank-r copy (SVD, r = round(rho * max rank)). Only the stages that use W are computed again.
Nothing is trained. The archive and the bundles are only read.

Output (in --out):
  laser_rows.csv      one row per (model, matrix, rho), plus one row 'original' per model;
                      F1 (%) and MCC on diag, guard and holdout.
  margins_M<n>.npz    margin (logit 1 - logit 0) of all 7,608 test samples for every row (float16).
  log.txt             time per model.
Finished models (margins file present) are skipped, so the script can be started again after a stop.
"""
import argparse
import csv
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from scalpel.engine import STAGE_INPUTS          # noqa: E402
from scalpel.models import build_model           # noqa: E402

MODELS = [0, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
BUNDLE_OF = {6: 'bundle_bert', 7: 'bundle_roberta', 15: 'bundle_codebert'}
RHOS = (0.01, 0.05, 0.1, 0.25, 0.5)
PARTS = {'diag': 0, 'guard': 1, 'holdout': 2}
FIELDS = ['model', 'param', 'type', 'branch', 'pos', 'rho', 'rank', 'full_rank'] + \
         ['%s_%s' % (p, m) for p in PARTS for m in ('f1', 'mcc')]


def kind_of(no, name):
    """(type, branch, layer) of a parameter, or None if it is not edited."""
    low = name.lower()
    if any(s in low for s in ('position', 'embedclass', 'token_type', 'layernorm', 'norm')):
        return None
    if no == 15:
        branch = 'head' if name.startswith('classifier') else 'joint'
    elif re.search(r'(^|\.)(classifier|fc\w*|combined_fc)\.', name):
        branch = 'head'
    elif 'Msg' in name:
        branch = 'msg'
    elif 'Twin' in name:
        branch = 'code'
    else:
        branch = 'shared'                             # used by all branches (CNN filters)
    m = re.search(r'(?:layers|layer)\.(\d+)\.', name)
    layer = int(m.group(1)) if m else None
    if 'embed' in low:
        t = 'embed'
    elif 'in_proj' in name or re.search(r'attention\.self\.(query|key|value)', name):
        t = 'attn_in'
    elif 'out_proj' in name or 'attention.output.dense' in name:
        t = 'attn_out'
    elif 'linear1' in name or 'intermediate.dense' in name:
        t = 'ffn_in'
    elif 'linear2' in name or name.endswith('output.dense.weight'):
        t = 'ffn_out'
    elif 'lstm' in low:
        t = 'rnn'
    elif 'conv' in low:
        t = 'conv'
    elif branch == 'head':
        t = 'head'
    else:
        t = 'other'
    return t, branch, layer


def candidates(no, model, bert_layers, skip_bert_qkv=True):
    """List of (name, type, branch, pos) of the matrices to edit."""
    out, layers = [], {}
    for name, p in model.named_parameters():
        if p.dim() < 2 or min(p.shape[0], int(np.prod(p.shape[1:]))) <= 2:
            continue                                  # vectors and the 2-row output layers
        k = kind_of(no, name)
        if k is None or not model.stage_of(name):
            continue
        t, branch, layer = k
        enc = re.sub(r'(?:layers|layer)\.\d+\..*', '', name)
        if layer is not None:
            layers[enc] = max(layers.get(enc, 0), layer)
        out.append((name, t, branch, layer, enc))
    res = []
    for name, t, branch, layer, enc in out:
        if layer is None:
            pos = '-'
        else:
            pos = 'last' if layer == layers[enc] else 'below%d' % (layers[enc] - layer)
        if no == 15 and layer is not None:
            if layers[enc] - layer >= bert_layers:
                continue                              # only the top layers of CodeBERT
            if skip_bert_qkv and t == 'attn_in':
                continue
        res.append((name, t, branch, pos))
    return res


def metrics(y, margin, idx):
    p = margin[idx] > 0
    t = y[idx].astype(bool)
    tp, fp = int(np.sum(p & t)), int(np.sum(p & ~t))
    fn, tn = int(np.sum(~p & t)), int(np.sum(~p & ~t))
    f1 = 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return f1, ((tp * tn - fp * fn) / d if d > 0 else 0.0)


class Runner:
    def __init__(self, no, model, b, dev, bs, amp):
        self.no, self.model, self.b, self.dev, self.bs, self.amp = no, model, b, dev, bs, amp
        self.n = len(b['y'])

    def stage(self, st):
        keys, outs = STAGE_INPUTS[st], []
        for i in range(0, self.n, self.bs):
            batch = {k: torch.from_numpy(np.ascontiguousarray(self.b[k][i:i + self.bs]).astype(np.int64)).to(self.dev)
                     for k in keys}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                # clone: the output can be a view of a much larger tensor
                outs.append(self.model.run_stage(st, batch, None).float().cpu().clone())
        return torch.cat(outs)

    def head(self, feats):
        outs = []
        for i in range(0, self.n, 1024):
            acts = {k: v[i:i + 1024].to(self.dev) for k, v in feats.items()}
            with torch.no_grad():
                outs.append(self.model.run_stage('head', None, acts).float().cpu())
        lg = torch.cat(outs).numpy().astype(np.float64)
        return lg[:, 1] - lg[:, 0]


def run_model(no, args, writer, fcsv, log):
    t0 = time.time()
    bdir = os.path.join(args.bundles, BUNDLE_OF.get(no, 'bundle'))
    z = np.load(os.path.join(bdir, 'bundle.npz'))
    keys = {'y', 'part', 'msg', 'tok_b', 'typ_b', 'tok_a', 'typ_a'}
    b = {k: z[k] for k in z.files if k in keys}
    if args.limit_rows:
        b = {k: v[:args.limit_rows] for k, v in b.items()}
    y, part = b['y'].astype(np.int64), b['part']
    idx = {p: np.where(part == v)[0] for p, v in PARTS.items()}
    dev = args.device
    model = build_model(args.archive, no).to(dev)
    amp = dev == 'cuda' and no == 15
    r = Runner(no, model, b, dev, args.batch_bert if no == 15 else args.batch, amp)
    stages = [s for s in model.STAGES if s != 'head']
    feats = {s: r.stage(s) for s in stages}
    m0 = r.head(feats)
    margins = {'original': m0.astype(np.float16)}

    def row(name, t, branch, pos, rho, rank, full, mg):
        d = {'model': no, 'param': name, 'type': t, 'branch': branch, 'pos': pos, 'rho': rho,
             'rank': rank, 'full_rank': full}
        for p in PARTS:
            f1, mcc = metrics(y, mg, idx[p])
            d[p + '_f1'], d[p + '_mcc'] = round(f1, 4), round(mcc, 5)
        writer.writerow(d)

    row('original', '-', '-', '-', 0, 0, 0, m0)
    cands = candidates(no, model, args.bert_layers)
    if args.max_params:
        cands = cands[:args.max_params]
    params = dict(model.named_parameters())
    for name, t, branch, pos in cands:
        p = params[name]
        w0 = p.data.clone()
        mat = w0.reshape(w0.shape[0], -1).float()
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        full = S.numel()
        st = model.stage_of(name)
        for rho in args.rhos:
            rank = max(1, int(round(rho * full)))
            p.data = ((U[:, :rank] * S[:rank]) @ Vh[:rank]).reshape(w0.shape).to(w0.dtype)
            f2 = dict(feats)
            for s in st:
                if s != 'head':
                    f2[s] = r.stage(s)
            mg = r.head(f2)
            margins['%s|%s' % (name, rho)] = mg.astype(np.float16)
            row(name, t, branch, pos, rho, rank, full, mg)
        p.data = w0
        del U, S, Vh
        fcsv.flush()
    np.savez_compressed(os.path.join(args.out, 'margins_M%d.npz' % no), **margins)
    msg = 'M%d: %d matrices x %d rho, %.0f s' % (no, len(cands), len(args.rhos), time.time() - t0)
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()
    del model
    if dev == 'cuda':
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', default=os.environ.get('SCALPEL_ARCHIVE', os.path.join(ROOT, '..', 'model-archive')))
    ap.add_argument('--bundles', default=ROOT, help='folder that holds bundle/, bundle_bert/, bundle_roberta/, bundle_codebert/')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--models', default=','.join(map(str, MODELS)))
    ap.add_argument('--rhos', default=','.join(map(str, RHOS)))
    ap.add_argument('--bert-layers', type=int, default=4, help='CodeBERT: edit only the top N encoder layers')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch', type=int, default=64, help='samples per pass (attention memory grows with heads x 800^2)')
    ap.add_argument('--batch-bert', type=int, default=32)
    ap.add_argument('--limit-rows', type=int, default=0, help='smoke test: first N samples only')
    ap.add_argument('--max-params', type=int, default=0, help='smoke test: first N matrices only')
    args = ap.parse_args()
    args.rhos = [float(x) for x in args.rhos.split(',')]
    os.makedirs(args.out, exist_ok=True)
    print('python', sys.version.split()[0], 'torch', torch.__version__, 'device', args.device, flush=True)
    path = os.path.join(args.out, 'laser_rows.csv')
    done = {int(f[9:-4]) for f in os.listdir(args.out) if f.startswith('margins_M') and f.endswith('.npz')}
    rows = []
    if os.path.exists(path):              # keep rows of finished models only
        with open(path, newline='') as f:
            rows = [x for x in csv.DictReader(f) if int(x['model']) in done]
    with open(path, 'w', newline='') as fcsv, open(os.path.join(args.out, 'log.txt'), 'a') as log:
        w = csv.DictWriter(fcsv, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
        fcsv.flush()
        for no in [int(x) for x in args.models.split(',')]:
            if no in done:
                print('M%d: done before, skipped' % no, flush=True)
                continue
            run_model(no, args, w, fcsv, log)
    print('finished:', path, flush=True)


if __name__ == '__main__':
    main()
