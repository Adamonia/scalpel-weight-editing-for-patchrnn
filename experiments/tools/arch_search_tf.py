"""Search for the TwinTransformer forward configuration that reproduces predicted.txt."""
import itertools
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/home/claude/work/scalpel_project')
from scalpel.data import build_dataset
from scalpel.models import load_state_dict

BASE = '/mnt/user-data/uploads/patchrnn-archive/model-archive/'


class Cand(nn.Module):
    def __init__(self, sd, nhead, activation, norm_first):
        super().__init__()
        d = sd['embedTwin.weight'].shape[1]
        ff = sd['transformerTwinB.layers.0.linear1.weight'].shape[0]
        n_layers = len({k.split('.')[2] for k in sd if k.startswith('transformerTwinB.layers.')})
        self.embedPositionTwin = nn.Parameter(torch.zeros_like(sd['embedPositionTwin']))
        self.embedPositionMsg = nn.Parameter(torch.zeros_like(sd['embedPositionMsg']))
        self.embedClassB = nn.Parameter(torch.zeros(1, 1, d))
        self.embedClassA = nn.Parameter(torch.zeros(1, 1, d))
        self.embedClassM = nn.Parameter(torch.zeros(1, 1, d))
        self.embedTwin = nn.Embedding(*sd['embedTwin.weight'].shape)
        self.embedMsg = nn.Embedding(*sd['embedMsg.weight'].shape)

        def enc(name):
            layer = nn.TransformerEncoderLayer(d, nhead, ff, dropout=0.0, activation=activation,
                                               batch_first=True, norm_first=norm_first)
            norm = nn.LayerNorm(d) if (name + '.norm.weight') in sd else None
            return nn.TransformerEncoder(layer, n_layers, norm=norm, enable_nested_tensor=False)
        self.transformerTwinB = enc('transformerTwinB')
        self.transformerTwinA = enc('transformerTwinA')
        self.transformerMsg = enc('transformerMsg')
        c = 3 * d
        self.classifier = nn.Sequential(nn.LayerNorm(c), nn.Identity(), nn.Linear(c, c), nn.LayerNorm(c), nn.Linear(c, 2))
        self.load_state_dict(sd)

    def branch(self, emb, pos, cls, encoder, tok, cls_mode, mask_pad, pool, scale):
        x = emb(tok.long())
        if scale:
            x = x * (x.shape[-1] ** 0.5)
        b = x.shape[0]
        c = cls.expand(b, -1, -1)
        pad = (tok == 0)
        if cls_mode == 'pre_after_pos':
            x = torch.cat([c, x + pos], 1)
            pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool), pad], 1)
        elif cls_mode == 'pre_trunc':
            x = torch.cat([c, x[:, :-1]], 1) + pos
            pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool), pad[:, :-1]], 1)
        elif cls_mode == 'post_after_pos':
            x = torch.cat([x + pos, c], 1)
            pad = torch.cat([pad, torch.zeros(b, 1, dtype=torch.bool)], 1)
        elif cls_mode == 'add':
            x = x + pos + c
        elif cls_mode == 'none':
            x = x + pos
        out = encoder(x, src_key_padding_mask=pad if mask_pad else None)
        if pool == 'first':
            return out[:, 0]
        if pool == 'last':
            return out[:, -1]
        if pool == 'mean':
            return out.mean(1)
        if pool == 'max':
            return out.max(1)[0]
        raise KeyError(pool)

    def feats(self, batch, cls_mode, mask_pad, pool, scale):
        fb = self.branch(self.embedTwin, self.embedPositionTwin, self.embedClassB, self.transformerTwinB, batch['tok_b'], cls_mode, mask_pad, pool, scale)
        fa = self.branch(self.embedTwin, self.embedPositionTwin, self.embedClassA, self.transformerTwinA, batch['tok_a'], cls_mode, mask_pad, pool, scale)
        fm = self.branch(self.embedMsg, self.embedPositionMsg, self.embedClassM, self.transformerMsg, batch['msg'], cls_mode, mask_pad, pool, scale)
        return torch.cat([fb, fa, fm], 1)

    def head(self, f, cls_act):
        f = self.classifier[0](f)
        f = {'id': lambda t: t, 'relu': F.relu, 'gelu': F.gelu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}[cls_act](f)
        f = self.classifier[2](f)
        f = self.classifier[3](f)
        return self.classifier[4](f)


def main():
    model_no = int(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    d = build_dataset('/home/claude/ext/pub_temp', '/home/claude/work/cache/dataset.npz')
    sd = load_state_dict(BASE + f'Model{model_no}/content/temp/model_TwinTransformer.pth')
    pf = np.array([int(x) for x in open(BASE + f'Model{model_no}/content/predicted.txt').read().split()])[:n]
    idx = d['test_idx'][:n]
    batch = {k: torch.from_numpy(d[k][idx]) for k in ['tok_b', 'tok_a', 'msg']}
    res = []
    t0 = time.time()
    stop = False
    for nhead, activation, norm_first in itertools.product([8, 4, 2, 16, 1], ['relu', 'gelu'], [False, True]):
        m = Cand(sd, nhead, activation, norm_first).eval()
        for cls_mode, mask_pad, pool, scale in itertools.product(
                ['pre_after_pos', 'pre_trunc', 'post_after_pos', 'add', 'none'], [False, True], ['first', 'last', 'mean'], [False]):
            if cls_mode == 'post_after_pos' and pool == 'first':
                continue
            if cls_mode in ('pre_after_pos', 'pre_trunc') and pool == 'last':
                continue
            with torch.no_grad():
                f = m.feats(batch, cls_mode, mask_pad, pool, scale)
                for cls_act in ['id', 'relu', 'gelu', 'tanh']:
                    out = m.head(f, cls_act)
                    agree = int((out.argmax(1).numpy() == pf).sum())
                    res.append((agree, nhead, activation, norm_first, cls_mode, mask_pad, pool, scale, cls_act))
        grp = sorted([r for r in res if r[1:4] == (nhead, activation, norm_first)], reverse=True)
        print('done', nhead, activation, norm_first, round(time.time() - t0), 's best', [(g[0],) + g[4:] for g in grp[:6]], flush=True)
        if grp and grp[0][0] >= n - 1:
            break
    res.sort(reverse=True)
    for r in res[:12]:
        print(r)
    print('n =', n, 'worst', res[-1][0], 'median', res[len(res) // 2][0])


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
