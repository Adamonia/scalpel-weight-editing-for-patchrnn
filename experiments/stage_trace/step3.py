"""Step 3 (S4): weight norms of the message branch and the classifier, six models, from the checkpoints.
Writes out/weights.json. Spectral norm = largest singular value. Frobenius and spectral norms of in_proj are
given per block (Q, K, V). LayerNorm gain = mean of the weight vector. The forward code applies no scale
factor to the embeddings (checked in scalpel/models.py run_stage: embedding + position, no multiplication).
"""
import json
from common import *

out = {}
for no in MODELS_H:
    s = get_session(no)
    m = s.model
    sd = {k: v.detach().float() for k, v in m.state_dict().items()}
    d = sd['embedMsg.weight'].shape[1]

    def nrm(w):
        w = w.reshape(w.shape[0], -1)
        return {'fro': float(torch.linalg.matrix_norm(w, 'fro')), 'spec': float(torch.linalg.matrix_norm(w, 2))}
    r = {'n_heads_registry': int(m.nhead), 'n_layers': len(m.transformerMsg.layers), 'layers': []}
    for li in range(r['n_layers']):
        p = 'transformerMsg.layers.%d.' % li
        Wi = sd[p + 'self_attn.in_proj_weight']
        r['layers'].append({'Q': nrm(Wi[:d]), 'K': nrm(Wi[d:2 * d]), 'V': nrm(Wi[2 * d:]), 'out_proj': nrm(sd[p + 'self_attn.out_proj.weight']),
                            'linear1': nrm(sd[p + 'linear1.weight']), 'linear2': nrm(sd[p + 'linear2.weight']),
                            'norm1_gain_mean': float(sd[p + 'norm1.weight'].mean()), 'norm2_gain_mean': float(sd[p + 'norm2.weight'].mean()),
                            'in_proj_bias_V_norm': float(sd[p + 'self_attn.in_proj_bias'][2 * d:].norm())})
    r['final_norm_gain_mean'] = float(sd['transformerMsg.norm.weight'].mean())
    r['classifier'] = {'linear_2 (384x384)': nrm(sd['classifier.2.weight']), 'linear_4 (2x384)': nrm(sd['classifier.4.weight']),
                       'ln0_gain_mean': float(sd['classifier.0.weight'].mean()), 'ln3_gain_mean': float(sd['classifier.3.weight'].mean()),
                       'margin_vector_norm (|w1-w0| of linear_4)': float((sd['classifier.4.weight'][1] - sd['classifier.4.weight'][0]).norm())}
    # msg part of the classifier input: columns 256..383 of linear_2 and of LN0 gain
    W2 = sd['classifier.2.weight']
    r['classifier']['linear_2_msg_columns'] = nrm(W2[:, 256:])
    r['classifier']['linear_2_twin_columns'] = nrm(W2[:, :256])
    r['classifier']['ln0_gain_mean_msg_part'] = float(sd['classifier.0.weight'][256:].mean())
    r['classifier']['ln0_gain_mean_twin_part'] = float(sd['classifier.0.weight'][:256].mean())
    U = untrained_ids(s)
    E = sd['embedMsg.weight']
    mask = torch.zeros(E.shape[0], dtype=torch.bool); mask[torch.from_numpy(U)] = True
    mask[s.pad_id] = False
    pool = torch.from_numpy(trained_pool(s))
    r['embedding'] = {'trained_rows_mean_norm': float(E[pool].norm(dim=1).mean()), 'n_trained_rows': int(len(pool)),
                      'untrained_rows_mean_norm_original': float(E[torch.from_numpy(U)].norm(dim=1).mean()),
                      'trained_rows_mean_abs_entry': float(E[pool].abs().mean()),
                      'position_embedding_mean_row_norm': float(sd['embedPositionMsg'].reshape(-1, d).norm(dim=1).mean()),
                      'cls_token_norm': float(sd['embedClassM'].norm()), 'scale_factor_on_embeddings': 'none in forward code'}
    out['M%d' % no] = r
json.dump(out, open('%s/weights.json' % OUT, 'w'), indent=1)
print('ok')
