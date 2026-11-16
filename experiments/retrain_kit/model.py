"""TwinTransformer, copied from the scalpel project (models.py) and reduced to what the kit needs.

The state-dict keys and shapes are the same as in the archive checkpoints.
Embeddings keep the default nn.Embedding init N(0,1). Do not change this:
the study needs untrained rows to stay at N(0,1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TWIN_MAX_LEN = 800
MSG_MAX_LEN = 200
D_MODEL = 128
D_FF = 512
HEAD_DROPOUT = 0.5

# name -> (heads, layers). Every other size comes from the archive checkpoints.
# The six old names are archive models. The four new ones fill the grid heads {1,2,4,8,16} x layers {1,2}.
# A second layer is built the same way for all names: nn.TransformerEncoder(num_layers=2), one final LayerNorm.
ARCHS = {'M3': (1, 1), 'M4': (2, 1), 'M5': (8, 1), 'M8': (16, 1), 'M13': (1, 2), 'M12': (8, 2),
         'H4L1': (4, 1), 'H2L2': (2, 2), 'H4L2': (4, 2), 'H16L2': (16, 2)}


def one_hot_types(types):
    """Token types 1..5 become 5 binary features. Type 0 (padding) gives zeros."""
    return F.one_hot(types.long(), num_classes=6)[..., 1:].float()


class TwinTransformer(nn.Module):
    def __init__(self, v_twin, v_msg, nhead, n_layers, dropout=0.2):
        super().__init__()
        d = D_MODEL
        self.embedTwin = nn.Embedding(v_twin, d)
        self.embedPositionTwin = nn.Parameter(torch.randn(1, TWIN_MAX_LEN, d))
        self.embedPositionMsg = nn.Parameter(torch.randn(1, MSG_MAX_LEN, d))
        self.embedMsg = nn.Embedding(v_msg, d)
        self.embedClassB = nn.Parameter(torch.randn(1, 1, d))
        self.embedClassA = nn.Parameter(torch.randn(1, 1, d))
        self.embedClassM = nn.Parameter(torch.randn(1, 1, d))

        def encoder(final_norm):
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=D_FF,
                                               dropout=dropout, activation='gelu', batch_first=True)
            norm = nn.LayerNorm(d) if final_norm else None
            return nn.TransformerEncoder(layer, num_layers=n_layers, norm=norm,
                                         enable_nested_tensor=False, mask_check=False)

        self.transformerTwinB = encoder(True)
        self.transformerTwinA = encoder(False)
        self.transformerMsg = encoder(True)
        c = 3 * d
        self.classifier = nn.Sequential(nn.LayerNorm(c), nn.Dropout(p=HEAD_DROPOUT), nn.Linear(c, c),
                                        nn.LayerNorm(c), nn.Linear(c, 2))
        self.nhead = nhead

    def code_feature(self, tok, typ, which):
        """Feature of one code branch. which is 'B' (before) or 'A' (after)."""
        cls = self.embedClassB if which == 'B' else self.embedClassA
        enc = self.transformerTwinB if which == 'B' else self.transformerTwinA
        tok = tok[:, :TWIN_MAX_LEN - 1].long()
        typ = typ[:, :TWIN_MAX_LEN - 1]
        # The 5 token-type features replace the last 5 embedding dimensions.
        x = torch.cat((self.embedTwin(tok)[:, :, :-5], one_hot_types(typ)), dim=2)
        x = torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1)
        return enc(self.embedPositionTwin + x)[:, 0]

    def msg_feature(self, msg):
        x = self.embedMsg(msg[:, :MSG_MAX_LEN - 1].long())
        x = torch.cat((self.embedClassM.expand(x.shape[0], -1, -1), x), dim=1)
        return self.transformerMsg(self.embedPositionMsg + x)[:, 0]

    def head(self, fb, fa, fm):
        return self.classifier(torch.cat((fb, fa, fm), dim=1))

    def forward(self, tok_b, typ_b, tok_a, typ_a, msg):
        return self.head(self.code_feature(tok_b, typ_b, 'B'), self.code_feature(tok_a, typ_a, 'A'),
                         self.msg_feature(msg))


def build_arch(name, v_twin, v_msg, dropout=0.2):
    nhead, n_layers = ARCHS[name]
    return TwinTransformer(v_twin, v_msg, nhead, n_layers, dropout)


def arch_table():
    """List of dicts {arch, heads, layers} in ARCHS order."""
    return [{'arch': a, 'heads': h, 'layers': n} for a, (h, n) in ARCHS.items()]
