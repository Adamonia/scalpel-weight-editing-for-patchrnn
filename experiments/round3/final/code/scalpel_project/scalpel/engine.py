"""Session engine.

A session holds one archive model, the test data, the cached stage outputs,
the weight edits, the journal, and the sealed holdout part.

State on disk (run folder):
  config.json            paths and limits
  journal.jsonl          one line for each tool call, edit, commit and rollback
  weights_committed.pt   tensors that the committed edits changed
  weights_working.pt     tensors that the pending edits changed
  feats_committed.npz    stage outputs for the committed weights
  feats_working.npz      stage outputs for the working weights
  FINAL.json             written by finalize(); the session is then locked
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from .data import MSG_MAX_LEN, TWIN_MAX_LEN
from .models import REGISTRY, TwinTransformer, build_model, checkpoint_path

PARTS = {'diag': 0, 'guard': 1, 'holdout': 2}
STAGE_INPUTS = {'twin_b': ('tok_b', 'typ_b'), 'twin_a': ('tok_a', 'typ_a'), 'msg': ('msg',)}
DEFAULT_LIMITS = {'max_commits': 40, 'max_evaluations': 150, 'guard_metric': 'mcc'}     # guard_metric: mcc or f1


class ScalpelError(Exception):
    """An error that the agent can read and correct."""


def confusion_metrics(y, pred):
    """Confusion counts and the usual metrics. The positive class is 1 (security patch)."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    mcc = (tp * tn - fp * fn) / den ** 0.5 if den > 0 else 0.0
    return {'n': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'acc': round(100.0 * (tp + tn) / n, 3) if n else 0.0,
            'precision': round(100.0 * prec, 3), 'recall': round(100.0 * rec, 3),
            'f1': round(100.0 * f1, 3), 'mcc': round(mcc, 5)}


class Session:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, 'config.json'), encoding='utf-8') as fp:
            self.cfg = json.load(fp)
        self.model_no = self.cfg['model_no']
        self.label = REGISTRY[self.model_no][3]
        self.batch_size = self.cfg.get('batch_size', 64)
        self.limits = dict(DEFAULT_LIMITS, **self.cfg.get('limits', {}))

        z = np.load(os.path.join(self.cfg['bundle_dir'], 'bundle.npz'))
        self.b = {k: z[k] for k in z.files}
        self.y = self.b['y'].astype(np.int64)
        # Padding token id of the inputs (0 for the archive vocabularies, 1 for RoBERTa sub-words).
        self.pad_id = int(self.b['pad_id']) if 'pad_id' in self.b else 0
        self.part = self.b['part']
        self.n = len(self.y)
        with open(os.path.join(self.cfg['bundle_dir'], 'vocab.json'), encoding='utf-8') as fp:
            self.vocab = json.load(fp)
        self._tok_index = {}
        self._raw = None

        self.model = build_model(self.cfg['archive_dir'], self.model_no)
        self.is_tf = isinstance(self.model, TwinTransformer)
        self.win_code = TWIN_MAX_LEN - 1 if self.is_tf else TWIN_MAX_LEN
        self.win_msg = MSG_MAX_LEN - 1 if self.is_tf else MSG_MAX_LEN
        if hasattr(self.model, 'WINDOW'):
            self.win_code, self.win_msg = self.model.WINDOW
        sfx = '_tf' if self.is_tf else ''
        self.df = {
            'msg': (self.b['df_msg%s_pos' % sfx], self.b['df_msg%s_neg' % sfx]),
            'code': (self.b['df_code%s_pos' % sfx], self.b['df_code%s_neg' % sfx]),
        }
        self.trained = {t: (self.df[t][0] + self.df[t][1]) > 0 for t in ('msg', 'code')}
        # Embedding tensors of each table. A hybrid model has one tensor for each sub-model.
        self.table_tensors = getattr(self.model, 'TABLES', {'msg': ('embedMsg.weight',), 'code': ('embedTwin.weight',)})
        self.table_param = {t: names[0] for t, names in self.table_tensors.items() if names}
        self.base = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.params = dict(self.model.named_parameters())

        self.committed_delta = {}
        self.working_touched = set()
        self.pending = []            # Edit records that are not committed.
        self.commits = []            # Committed transactions.
        self.n_evaluations = 0
        self.final = None
        self._load_state()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def create(run_dir, archive_dir, bundle_dir, model_no, limits=None, batch_size=None):
        if batch_size is None:
            # Attention maps of 800 x 800 for each head use much memory.
            cls = REGISTRY[model_no][1]
            batch_size = getattr(cls, 'BATCH', 16 if cls.__name__ in ('TwinTransformer', 'TwinComb3') else 64)
        os.makedirs(run_dir, exist_ok=True)
        cfg_path = os.path.join(run_dir, 'config.json')
        if os.path.exists(cfg_path):
            raise ScalpelError('The run folder has a session already: ' + run_dir)
        cfg = {'archive_dir': archive_dir, 'bundle_dir': bundle_dir, 'model_no': model_no,
               'batch_size': batch_size, 'limits': limits or {}, 'created': time.time()}
        with open(cfg_path, 'w', encoding='utf-8') as fp:
            json.dump(cfg, fp, indent=1)
        s = Session(run_dir)
        s.log('session_created', {'model_no': model_no, 'label': s.label,
                                  'checkpoint_sha256': s.checkpoint_hash()})
        return s

    def checkpoint_hash(self):
        h = hashlib.sha256()
        with open(checkpoint_path(self.cfg['archive_dir'], self.model_no), 'rb') as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------ persistence
    def _p(self, name):
        return os.path.join(self.run_dir, name)

    def _load_state(self):
        base_feats = os.path.join(self.cfg['bundle_dir'], 'base_feats_M%d.npz' % self.model_no)
        if not os.path.exists(base_feats):
            feats = {}
            rows = np.arange(self.n)
            for st in self.model.STAGES[:-1]:
                feats[st] = self._compute_stage(st, rows)
            feats['head'] = self._compute_head(feats, rows)
            np.savez(base_feats, **feats)
        z = np.load(base_feats)
        self.base_feats = {k: z[k] for k in z.files}
        self.base_pred = self.base_feats['head'].argmax(1)

        if os.path.exists(self._p('state.json')):
            with open(self._p('state.json'), encoding='utf-8') as fp:
                st = json.load(fp)
            self.pending = st['pending']
            self.commits = st['commits']
            self.n_evaluations = st['n_evaluations']
            self.working_touched = set(st['working_touched'])
            if os.path.exists(self._p('weights_committed.pt')):
                self.committed_delta = torch.load(self._p('weights_committed.pt'), weights_only=True)
            for k, v in self.committed_delta.items():
                self.params[k].data.copy_(v)
            if os.path.exists(self._p('weights_working.pt')):
                for k, v in torch.load(self._p('weights_working.pt'), weights_only=True).items():
                    self.params[k].data.copy_(v)
            self.feats, self.valid = self._load_feats('feats_working.npz')
            self.c_feats, self.c_valid = self._load_feats('feats_committed.npz')
        else:
            self.feats = {k: v.copy() for k, v in self.base_feats.items()}
            self.valid = {k: np.ones(self.n, dtype=bool) for k in self.model.STAGES}
            self.c_feats = {k: v.copy() for k, v in self.feats.items()}
            self.c_valid = {k: v.copy() for k, v in self.valid.items()}
        if os.path.exists(self._p('FINAL.json')):
            with open(self._p('FINAL.json'), encoding='utf-8') as fp:
                self.final = json.load(fp)

    def _load_feats(self, name):
        z = np.load(self._p(name))
        feats = {k: z[k] for k in self.model.STAGES}
        valid = {k: z['valid_' + k] for k in self.model.STAGES}
        return feats, valid

    def save(self):
        st = {'pending': self.pending, 'commits': self.commits, 'n_evaluations': self.n_evaluations,
              'working_touched': sorted(self.working_touched)}
        with open(self._p('state.json'), 'w', encoding='utf-8') as fp:
            json.dump(st, fp)
        torch.save(self.committed_delta, self._p('weights_committed.pt'))
        torch.save({k: self.params[k].data.clone() for k in self.working_touched},
                   self._p('weights_working.pt'))
        for name, feats, valid in (('feats_working.npz', self.feats, self.valid),
                                   ('feats_committed.npz', self.c_feats, self.c_valid)):
            arrays = dict(feats)
            arrays.update({'valid_' + k: v for k, v in valid.items()})
            np.savez(self._p(name), **arrays)

    def log(self, kind, payload):
        rec = {'t': round(time.time(), 2), 'kind': kind}
        rec.update(payload)
        with open(self._p('journal.jsonl'), 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # ------------------------------------------------------------ computation
    def _batch(self, stage, rows, override=None):
        batch = {}
        for key in STAGE_INPUTS[stage]:
            arr = self.b[key][rows] if override is None or key not in override else override[key]
            batch[key] = torch.from_numpy(np.ascontiguousarray(arr))
        return batch

    def _compute_stage(self, stage, rows, override=None, n_rows=None):
        """Run one branch stage. 'override' can give input arrays in place of the test rows."""
        n_rows = len(rows) if n_rows is None else n_rows
        out = []
        with torch.no_grad():
            for s in range(0, n_rows, self.batch_size):
                sl = slice(s, s + self.batch_size)
                if override is None:
                    batch = self._batch(stage, rows[sl])
                else:
                    batch = {k: torch.from_numpy(np.ascontiguousarray(v[sl])) for k, v in override.items()}
                # Copy: the stage output can be a view of a much larger tensor.
                out.append(self.model.run_stage(stage, batch, None).numpy().copy())
        return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)

    def head_from(self, feat_b, feat_a, feat_m):
        """Logits from three feature arrays (numpy)."""
        acts = {'twin_b': torch.from_numpy(np.ascontiguousarray(feat_b)),
                'twin_a': torch.from_numpy(np.ascontiguousarray(feat_a)),
                'msg': torch.from_numpy(np.ascontiguousarray(feat_m))}
        with torch.no_grad():
            return self.model.run_stage('head', None, acts).numpy()

    def _compute_head(self, feats, rows):
        return self.head_from(feats['twin_b'][rows], feats['twin_a'][rows], feats['msg'][rows])

    def rows_of(self, part):
        if part == 'holdout' and self.final is None and not getattr(self, '_holdout_open', False):
            raise ScalpelError('The holdout part is sealed. Only finalize() reads it.')
        if part == 'dev':
            return np.where(self.part < 2)[0]
        if part not in PARTS:
            raise ScalpelError("Unknown part '%s'. Use diag, guard or dev." % part)
        return np.where(self.part == PARTS[part])[0]

    def refresh(self, rows):
        """Recompute the stale stage outputs for the given test rows."""
        for st in self.model.STAGES[:-1]:
            stale = rows[~self.valid[st][rows]]
            if len(stale):
                self.feats[st][stale] = self._compute_stage(st, stale)
                self.valid[st][stale] = True
        self.feats['head'][rows] = self._compute_head(self.feats, rows)
        self.valid['head'][rows] = True

    def logits(self, part):
        rows = self.rows_of(part)
        self.refresh(rows)
        return rows, self.feats['head'][rows]

    def metrics(self, part, committed=False):
        rows = self.rows_of(part)
        if committed:
            head = self.c_feats['head'][rows]
        else:
            self.refresh(rows)
            head = self.feats['head'][rows]
        return confusion_metrics(self.y[rows], head.argmax(1))

    # ------------------------------------------------------------------ edits
    def check_open(self):
        if self.final is not None:
            raise ScalpelError('The session is final. No more edits are possible.')

    def tensor(self, name):
        if name not in self.params:
            raise ScalpelError("Unknown tensor '%s'. Call overview to see the names." % name)
        return self.params[name].data

    def token_id(self, table, token):
        if table not in self._tok_index:
            self._tok_index[table] = {t: i for i, t in enumerate(self.vocab[table])}
        idx = self._tok_index[table].get(token)
        if idx is None:
            raise ScalpelError("Token '%s' is not in the %s vocabulary." % (token, table))
        return idx

    def rows_with_tokens(self, table, ids):
        """Test rows whose model input contains one of the token ids."""
        ids = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
        if table == 'msg':
            hit = np.isin(self.b['msg'][:, :self.win_msg], ids).any(1)
            return {'msg': hit}
        hit_b = np.isin(self.b['tok_b'][:, :self.win_code], ids).any(1)
        hit_a = np.isin(self.b['tok_a'][:, :self.win_code], ids).any(1)
        return {'twin_b': hit_b, 'twin_a': hit_a}

    def mark_stale(self, name, token_ids=None):
        """Mark the stage outputs that an edit of tensor 'name' makes stale."""
        stages = self.model.stage_of(name)
        table = next((t for t, names in self.table_tensors.items() if name in names), None)
        if table is not None and token_ids is not None:
            hits = self.rows_with_tokens(table, token_ids)
            for st in stages:
                if st in hits:
                    self.valid[st][hits[st]] = False
        else:
            for st in stages:
                if st != 'head':
                    self.valid[st][:] = False
        self.valid['head'][:] = False
        self.working_touched.add(name)

    def record_edit(self, rec):
        self.check_open()
        rec = dict(rec, seq=len(self.pending) + 1)
        self.pending.append(rec)
        self.log('edit', rec)
        return rec

    def rollback(self):
        self.check_open()
        n = len(self.pending)
        for name in self.working_touched:
            src = self.committed_delta.get(name, self.base[name])
            self.params[name].data.copy_(src)
        self.working_touched = set()
        self.pending = []
        self.feats = {k: v.copy() for k, v in self.c_feats.items()}
        self.valid = {k: v.copy() for k, v in self.c_valid.items()}
        self.log('rollback', {'n_edits_dropped': n})
        return n

    def evaluate(self):
        """Effect of the pending edits on the diag and guard parts."""
        self.n_evaluations += 1
        out = {}
        for part in ('diag', 'guard'):
            rows = self.rows_of(part)
            self.refresh(rows)
            new = self.feats['head'][rows].argmax(1)
            old = self.c_feats['head'][rows].argmax(1)
            y = self.y[rows]
            fixed = rows[(old != y) & (new == y)]
            broken = rows[(old == y) & (new != y)]
            out[part] = {'before': confusion_metrics(y, old), 'after': confusion_metrics(y, new),
                         'n_fixed': int(len(fixed)), 'n_broken': int(len(broken))}
            if part == 'diag':
                out[part]['fixed_ids'] = [int(r) for r in fixed[:40]]
                out[part]['broken_ids'] = [int(r) for r in broken[:40]]
        out['evaluations_used'] = self.n_evaluations
        return out

    def commit(self, rationale, error_type):
        self.check_open()
        if not self.pending:
            raise ScalpelError('There are no pending edits.')
        if len(self.commits) >= self.limits['max_commits']:
            raise ScalpelError('The commit limit is reached. Call finalize.')
        ev = self.evaluate()
        key = self.limits.get('guard_metric', 'mcc')
        for part in ('diag', 'guard'):
            if ev[part]['after'][key] < ev[part]['before'][key] - 1e-9:
                raise ScalpelError(
                    'Commit refused: the %s of the %s part decreases (%.5f -> %.5f). '
                    'Change the edit or call rollback.' % (key.upper(), part, ev[part]['before'][key], ev[part]['after'][key]))
        for name in self.working_touched:
            self.committed_delta[name] = self.params[name].data.clone()
        rec = {'commit': len(self.commits) + 1, 'rationale': rationale, 'error_type': error_type,
               'edits': self.pending, 'diag': ev['diag'], 'guard': ev['guard']}
        self.commits.append(rec)
        self.pending = []
        self.working_touched = set()
        self.c_feats = {k: v.copy() for k, v in self.feats.items()}
        self.c_valid = {k: v.copy() for k, v in self.valid.items()}
        self.log('commit', rec)
        return rec

    # --------------------------------------------------------------- raw text
    def raw(self, row):
        if self._raw is None:
            path = os.path.join(self.cfg['bundle_dir'], 'raw_test.json')
            with open(path, encoding='utf-8') as fp:
                self._raw = json.load(fp)
        return self._raw[row]

    def visible(self, row):
        """Only diag samples can be shown one by one."""
        row = int(row)
        if row < 0 or row >= self.n:
            raise ScalpelError('Sample id out of range.')
        if self.part[row] != PARTS['diag']:
            raise ScalpelError('Sample %d is not in the diag part. Only diag samples are readable.' % row)
        return row
