"""Train the Transformer architectures with several seeds. One run = one (architecture, seed).

Example: python train_seeds.py --data dataset.npz --out runs
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (VAL_SPLIT_SEED, f1_score, fit_val_rows, forward_batch, load_data, make_batch, pick_device,
                    read_json)
from model import ARCHS, MSG_MAX_LEN, build_arch
from unk_tools import add_unk_row, apply_map, build_unk_map

# ---------------------------------------------------------------------------
# DEFAULTS. Recipe of the public PatchRNN code (TwinRNNTrain in SecurityPatchIdentificationRNN.py):
# Adam, lr 0.0005, batch 256, no weight decay, no shuffling, up to 20 epochs, model selection by
# validation ACCURACY, stop when epoch >= 10 and the last accuracy is below the minimum of the 10 before.
# The Transformer variants may have used other values. This is not known.
# Dropout is not in that code. The values below are the ones of the scalpel model class.
EPOCHS = 20
LR = 5e-4
BATCH = 256
DROPOUT = 0.2          # encoder dropout. The classifier dropout stays 0.5.
WEIGHT_DECAY = 0.0     # Applied to weight matrices only. Embeddings never get decay (see README).
JUDGE_EPOCHS = 10
# ---------------------------------------------------------------------------


def parse_list(text, allowed=None):
    """'0-4' -> [0,1,2,3,4]. '0,2,5' -> [0,2,5]. 'M3,M4' -> ['M3','M4']."""
    out = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part and part.replace('-', '').isdigit():
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def untrained_msg_rows(msg, fit_rows, vocab_size):
    """Message token ids that never occur in the input window of a fit sample. The pad id 0 is excluded."""
    seen = np.zeros(vocab_size, dtype=bool)
    seen[np.unique(msg[fit_rows][:, :MSG_MAX_LEN - 1])] = True
    seen[0] = True
    return np.where(~seen)[0]


def make_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if name.startswith('embed') or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    return torch.optim.Adam(groups, lr=lr)


def evaluate(model, data, rows, device, use_amp, batch=64):
    model.eval()
    margins = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            b = make_batch(data, rows[s:s + batch], device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = forward_batch(model, b)
            logits = logits.float().cpu().numpy()
            margins.append(logits[:, 1] - logits[:, 0])
    m = np.concatenate(margins)
    # Stop if the model gives values that are not finite. This shows a numeric fault.
    if not np.isfinite(m).all():
        raise RuntimeError('model output is not finite (NaN or Inf)')
    y = data['labels'][rows]
    return f1_score(y, m), 100.0 * float(np.mean((m > 0) == (y == 1))), 100.0 * float(np.mean(m > 0))


def train_one(arch, seed, data, args, device):
    run_dir = Path(args.out) / ('%s_seed%d' % (arch, seed))
    log_path = run_dir / 'log.json'
    if log_path.exists():
        # A folder must not mix runs with and without UNK.
        if read_json(log_path)['args'].get('unk') != args.unk:
            raise SystemExit('%s was made with another --unk value. Use another --out folder.' % run_dir)
        print('skip (finished):', run_dir)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    # Mixed precision is off by default. The original code used full precision.
    use_amp = device == 'cuda' and args.amp
    fit_rows, val_rows = fit_val_rows(data, args.select_on, args.limit)
    y = data['labels']
    n_pos = int((y[fit_rows] == 1).sum())
    n_neg = int((y[fit_rows] == 0).sum())
    # Weight of class c is 1 - share of c in the training labels (as in the original code).
    weights = [1 - n_neg / (n_pos + n_neg), 1 - n_pos / (n_pos + n_neg)]
    class_weight = torch.tensor(weights, device=device)

    v_twin, v_msg = int(data['n_code_vocab']), int(data['n_msg_vocab'])
    unk_log = {}
    if args.unk is not None:
        # Every message token with c <= K goes to one new UNK id (train, val and test). Pad stays 0.
        table, unk_id, count = build_unk_map(data['msg'], fit_rows, v_msg, args.unk)
        data = apply_map(data, table)
        win = data['msg'][:, :MSG_MAX_LEN - 1]
        unk_log = {'unk_id': unk_id, 'n_ids_mapped': int((table[:v_msg] == unk_id).sum()),
                   'unk_share_fit_tokens': float((win[fit_rows] == unk_id).sum() / (win[fit_rows] != 0).sum()),
                   'unk_share_test_tokens': float((win[data['test_idx']] == unk_id).sum() / (win[data['test_idx']] != 0).sum())}
        print('UNK: K=%d, id %d, %d ids mapped' % (args.unk, unk_id, unk_log['n_ids_mapped']), flush=True)

    seed_everything(seed)
    model = build_arch(arch, v_twin, v_msg, args.dropout).to(device)
    if args.unk is not None:
        add_unk_row(model, seed)     # after build_arch: all other init values stay the same as without UNK
        v_msg += 1
    opt = make_optimizer(model, args.lr, args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    untrained = untrained_msg_rows(data['msg'], fit_rows, v_msg)
    init_rows = model.embedMsg.weight.detach()[torch.from_numpy(untrained).to(device)].clone()
    if args.unk is not None:
        unk_init = model.embedMsg.weight.detach()[unk_id].clone()

    rng = np.random.RandomState(seed)
    epochs, best_f1, best_epoch, step = [], -1.0, -1, 0
    acc_list = [0]     # Same start value as the original code.
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(fit_rows) if args.shuffle else fit_rows
        losses = []
        for s in range(0, len(order), args.batch):
            rows = order[s:s + args.batch]
            target_all = torch.from_numpy(y[rows].astype(np.int64)).to(device)
            # Weighted mean over the whole batch = sum of weighted losses / sum of weights.
            weight_sum = class_weight[target_all].sum()
            step += 1
            opt.zero_grad(set_to_none=True)
            total = 0.0
            # The batch is cut into micro-batches (gradient accumulation) to save memory.
            # The result is the same as one pass over the whole batch.
            for m in range(0, len(rows), args.micro_batch):
                b = make_batch(data, rows[m:m + args.micro_batch], device)
                target = target_all[m:m + args.micro_batch]
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    logits = forward_batch(model, b)
                loss = F.cross_entropy(logits.float(), target, weight=class_weight, reduction='sum') / weight_sum
                scaler.scale(loss).backward()
                total += loss.item()
            scaler.step(opt)
            scaler.update()
            losses.append(total)
        val_f1, val_acc, val_pos = evaluate(model, data, val_rows, device, use_amp, args.micro_batch)
        epochs.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_acc': val_acc, 'val_f1': val_f1,
                       'seconds': round(time.time() - t0, 1)})
        print('%s seed %d epoch %d loss %.4f val_acc %.2f val_f1 %.2f predicted_pos %.1f%% %.0fs' % (
            arch, seed, epoch, epochs[-1]['train_loss'], val_acc, val_f1, val_pos, epochs[-1]['seconds']), flush=True)
        acc_list.append(val_acc)
        # Save the model when the validation accuracy is a new best (rule of the original code).
        if acc_list[-1] > max(acc_list[:-1]):
            best_f1, best_epoch = val_f1, epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / 'model.pth')
        # Stop rule of the original code. epoch - 1 is the original 0-based epoch.
        if (epoch - 1) >= JUDGE_EPOCHS and acc_list[-1] < min(acc_list[-1 - JUDGE_EPOCHS:-1]):
            break

    # Check that the untrained message rows were never changed by training.
    change = (model.embedMsg.weight.detach()[torch.from_numpy(untrained).to(device)] - init_rows).abs().max().item()
    if args.unk is not None:
        unk_log['unk_row_max_change'] = (model.embedMsg.weight.detach()[unk_id] - unk_init).abs().max().item()
    log = {**unk_log, 'arch': arch, 'seed': seed, 'args': vars(args), 'epochs': epochs, 'best_epoch': best_epoch,
           'best_val_f1': best_f1, 'best_val_acc': max(acc_list), 'select_on': args.select_on, 'n_fit': len(fit_rows), 'n_val': len(val_rows), 'val_split_seed': VAL_SPLIT_SEED,
           'class_weights': weights, 'n_untrained_msg_rows': int(len(untrained)),
           'untrained_rows_max_change': change, 'seconds_total': round(time.time() - t_start, 1),
           'torch_version': torch.__version__, 'device': device}
    with open(log_path, 'w', encoding='utf-8') as fp:
        json.dump(log, fp, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='dataset.npz')
    p.add_argument('--archs', default=','.join(ARCHS))
    p.add_argument('--seeds', default='0-4')
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--lr', type=float, default=LR)
    p.add_argument('--batch', type=int, default=BATCH)
    p.add_argument('--dropout', type=float, default=DROPOUT)
    p.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--out', default='runs')
    p.add_argument('--micro-batch', type=int, default=16, help='samples per forward pass (memory only)')
    p.add_argument('--amp', action='store_true', help='mixed precision on cuda (faster, original: no)')
    p.add_argument('--shuffle', action='store_true', help='shuffle the training order (original: no)')
    p.add_argument('--select-on', default='val', choices=['val', 'test'],
                   help='val: 10 pct of train. test: replicates the original code (biased)')
    p.add_argument('--unk', type=int, default=None, metavar='K',
                   help='map every message token with training count c <= K to one UNK id (default: off)')
    p.add_argument('--limit', type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    device = pick_device(args.device)
    data = load_data(args.data)
    archs = parse_list(args.archs)
    seeds = parse_list(args.seeds)
    for a in archs:
        if a not in ARCHS:
            raise SystemExit('Unknown architecture %s. Use: %s' % (a, ', '.join(ARCHS)))
    print('device:', device, 'torch', torch.__version__)
    for seed in seeds:
        for arch in archs:
            train_one(arch, seed, data, args, device)


if __name__ == '__main__':
    main()
