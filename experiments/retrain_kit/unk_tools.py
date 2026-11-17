"""Helpers for the UNK experiment (train_seeds.py --unk K, analyse_unk.py).

Count c of a message token = number of FIT rows (the rows used for gradient training, not the validation rows)
that contain the token in the model window. This is token_counts() of apply_round3.py. We import it, so the
definition is the same everywhere.
"""
import numpy as np
import torch

from apply_round3 import PAD_ID, token_counts


def build_unk_map(msg, fit_rows, v_msg, k):
    """Return (table, unk_id, count).

    table has v_msg + 1 entries. table[t] = unk_id if c(t) <= k, else t. The pad id is never changed.
    unk_id = v_msg: one new row after the last id of the old vocabulary (the old ids are all in use or unknown).
    """
    count = token_counts(msg, fit_rows, v_msg)
    if count[PAD_ID] == 0:
        raise RuntimeError('pad id is not in the fit rows')
    unk_id = v_msg
    table = np.arange(v_msg + 1, dtype=np.int64)
    low = count <= k
    low[PAD_ID] = False
    table[:v_msg][low] = unk_id
    if table[PAD_ID] != PAD_ID or table[unk_id] != unk_id:
        raise RuntimeError('bad UNK table')
    if (table[:v_msg][low] != unk_id).any():
        raise RuntimeError('a token with c <= k was not mapped')
    return table, unk_id, count


def apply_map(data, table):
    """Copy of the data dict with the message tokens mapped. Code input is not changed."""
    out = dict(data)
    out['msg'] = table[data['msg'].astype(np.int64)].astype(np.int32)
    return out


def add_unk_row(model, seed):
    """Add one row to the message table. The row is N(0,1), the same init as the other rows.

    The row is made AFTER the model was built, with its own generator. So the init of every other weight
    and the later random numbers (dropout) are the same as in the run without UNK (same seed).
    """
    old = model.embedMsg.weight.data
    gen = torch.Generator().manual_seed(1000003 + int(seed))
    row = torch.randn(1, old.shape[1], generator=gen).to(old.device, old.dtype)
    new = torch.nn.Embedding(old.shape[0] + 1, old.shape[1]).to(old.device, old.dtype)
    new.weight.data = torch.cat((old, row), dim=0)
    model.embedMsg = new
    return old.shape[0]
