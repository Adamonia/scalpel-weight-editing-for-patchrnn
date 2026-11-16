"""How far is the mean trained message row from zero? (explains zero vs mean imputation)"""
import json
from common import *
out = {}
for no in MODELS:
    s = get_session(no)
    W = s.model.embedMsg.weight.detach()
    pool = torch.from_numpy(trained_pool(s))
    mean = W[pool].mean(0)
    out[no] = {'norm_mean_row': float(mean.norm()), 'mean_norm_of_trained_rows': float(W[pool].norm(dim=1).mean()),
               'ratio': float(mean.norm() / W[pool].norm(dim=1).mean()),
               'mean_norm_untrained_rows': float(W[torch.from_numpy(untrained_ids(s))].norm(dim=1).mean())}
json.dump(out, open(OUT + '/mean_vec_norm.json', 'w'), indent=1)
print(json.dumps(out))
