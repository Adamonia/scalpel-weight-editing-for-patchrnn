"""Model definitions for the archive checkpoints.

Each model gives the same interface:
  STAGES            names of the cached stages, in the order of the data flow.
  stage_of(param)   the stages that a parameter changes directly. An empty
                    tuple shows that the forward pass does not use the parameter.
  run_stage(...)    compute one stage from the inputs or from earlier stages.
The last stage is always 'head'. It returns the two logits.
The stages twin_b, twin_a and msg return one feature vector for each sample.
"""
import collections
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import MSG_MAX_LEN, TWIN_MAX_LEN


def load_state_dict(path):
    """Load a checkpoint. Only tensors are permitted (weights_only)."""
    return torch.load(path, map_location='cpu', weights_only=True)


def _one_hot_types(types):
    """Token types 1..5 become 5 binary features. Type 0 (padding) gives zeros."""
    return F.one_hot(types.long(), num_classes=6)[..., 1:].float()


class TwinRNN(nn.Module):
    """PatchRNN twin model (Wang et al., MILCOM 2021), as in the Colab notebook."""

    STAGES = ('twin_b', 'twin_a', 'msg', 'head')
    HEAD_LINEARS = ('fc1', 'fc3', 'fc4')
    OUT_BIAS = 'fc4.bias'
    DESCRIPTION = (
        'Inputs: before-version code tokens (800), after-version code tokens (800), message tokens (200). '
        'twin_b and twin_a: embedTwin row (128) + 5 token-type bits -> one shared bidirectional LSTM '
        '(lstmTwin, hidden 32). Feature = final hidden states of the two directions (64). '
        'The LSTM also reads the padding positions. '
        'msg: embedMsg row (128) -> bidirectional LSTM (lstmMsg, hidden 32) -> 64 features. '
        'head: fc1(cat(twin_b, twin_a)) [128->64] -> cat with msg -> fc3 [128->64] -> fc4 [64->2]. '
        'The head has no activation function, thus the margin is a linear function of the 192 branch features. '
        'fc2 is not used.')

    def __init__(self, sd):
        super().__init__()
        v_twin, e_twin = sd['embedTwin.weight'].shape
        v_msg, e_msg = sd['embedMsg.weight'].shape
        h_twin = sd['lstmTwin.weight_hh_l0'].shape[1]
        h_msg = sd['lstmMsg.weight_hh_l0'].shape[1]
        extra = sd['lstmTwin.weight_ih_l0'].shape[1] - e_twin
        self.embedTwin = nn.Embedding(v_twin, e_twin)
        self.lstmTwin = nn.LSTM(e_twin + extra, h_twin, num_layers=1, bidirectional=True)
        self.embedMsg = nn.Embedding(v_msg, e_msg)
        self.lstmMsg = nn.LSTM(e_msg, h_msg, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(h_twin * 4, h_twin * 2)
        self.fc2 = nn.Linear(h_twin * 2, 2)          # Not used when the message branch is on.
        self.fc3 = nn.Linear((h_twin + h_msg) * 2, h_twin + h_msg)
        self.fc4 = nn.Linear(h_twin + h_msg, 2)
        self.load_state_dict(sd)

    @staticmethod
    def stage_of(name):
        if name.startswith(('embedTwin', 'lstmTwin')):
            return ('twin_b', 'twin_a')
        if name.startswith(('embedMsg', 'lstmMsg')):
            return ('msg',)
        if name.startswith('fc2'):
            return ()
        return ('head',)

    def _twin(self, tok, typ):
        x = torch.cat((self.embedTwin(tok.long()), _one_hot_types(typ)), 2)
        _, (h_n, _) = self.lstmTwin(x.permute(1, 0, 2))
        return torch.cat([h_n[0], h_n[1]], dim=1)

    def run_stage(self, stage, batch, acts):
        if stage == 'twin_b':
            return self._twin(batch['tok_b'], batch['typ_b'])
        if stage == 'twin_a':
            return self._twin(batch['tok_a'], batch['typ_a'])
        if stage == 'msg':
            x = self.embedMsg(batch['msg'].long())
            _, (h_n, _) = self.lstmMsg(x.permute(1, 0, 2))
            return torch.cat([h_n[0], h_n[1]], dim=1)
        if stage == 'head':
            f = self.fc1(torch.cat((acts['twin_b'], acts['twin_a']), dim=1))
            f = self.fc3(torch.cat((f, acts['msg']), dim=1))
            return self.fc4(f)
        raise KeyError(stage)


class TwinCNN(nn.Module):
    """Conv2D-E model (Model2). Shared text convolutions and a linear head."""

    STAGES = ('twin_b', 'twin_a', 'msg', 'head')
    HEAD_LINEARS = ('fcB', 'fcA', 'fcM', 'fc', 'fc2')
    OUT_BIAS = 'fc2.bias'
    DESCRIPTION = (
        'Inputs: before-version code tokens (800), after-version code tokens (800), message tokens (200). '
        'twin_b, twin_a and msg: embedding rows (128) -> three shared banks of Conv2d filters (convs_1d, '
        '100 filters each, kernel heights 3, 4, 5, padding k-2) -> ReLU -> maximum over the positions -> '
        '300 features for each branch. The same filters read code and message. The filters also read the padding. '
        'head: fcB, fcA, fcM [300->150] -> cat (450) -> fc [450->150] -> fc2 [150->2]. '
        'The head has no activation function, thus the margin is a linear function of the 900 pooled features. '
        'norm (LayerNorm 300) is not used. Token types are not used.')

    def __init__(self, sd):
        super().__init__()
        self.embedTwin = nn.Embedding(*sd['embedTwin.weight'].shape)
        self.embedMsg = nn.Embedding(*sd['embedMsg.weight'].shape)
        self.convs_1d = nn.ModuleList(
            [nn.Conv2d(1, 100, (k, 128), padding=(k - 2, 0)) for k in (3, 4, 5)])
        self.norm = nn.LayerNorm(300)        # The forward pass does not use this layer.
        self.fcB = nn.Linear(300, 150)
        self.fcA = nn.Linear(300, 150)
        self.fcM = nn.Linear(300, 150)
        self.fc = nn.Linear(450, 150)
        self.fc2 = nn.Linear(150, 2)
        self.load_state_dict(sd)

    @staticmethod
    def stage_of(name):
        if name.startswith('embedTwin'):
            return ('twin_b', 'twin_a')
        if name.startswith('embedMsg'):
            return ('msg',)
        if name.startswith('convs_1d'):
            return ('twin_b', 'twin_a', 'msg')
        if name.startswith('norm'):
            return ()
        return ('head',)

    def _pool(self, emb):
        x = emb.unsqueeze(1)
        out = []
        for conv in self.convs_1d:
            c = F.relu(conv(x)).squeeze(3)
            out.append(F.max_pool1d(c, c.size(2)).squeeze(2))
        return torch.cat(out, 1)

    def run_stage(self, stage, batch, acts):
        if stage == 'twin_b':
            return self._pool(self.embedTwin(batch['tok_b'].long()))
        if stage == 'twin_a':
            return self._pool(self.embedTwin(batch['tok_a'].long()))
        if stage == 'msg':
            return self._pool(self.embedMsg(batch['msg'].long()))
        if stage == 'head':
            f = torch.cat((self.fcB(acts['twin_b']), self.fcA(acts['twin_a']),
                           self.fcM(acts['msg'])), 1)
            return self.fc2(self.fc(f))
        raise KeyError(stage)


class TwinTransformer(nn.Module):
    """Transformer-encoder twin model (Models 3, 4, 5, 8, 12, 13)."""

    STAGES = ('twin_b', 'twin_a', 'msg', 'head')
    HEAD_LINEARS = ('classifier.2', 'classifier.4')
    OUT_BIAS = 'classifier.4.bias'
    DESCRIPTION = (
        'Inputs: before-version code tokens (first 799), after-version code tokens (first 799), '
        'message tokens (first 199). '
        'twin_b and twin_a: embedTwin row with its LAST 5 dimensions replaced by the 5 token-type bits; '
        'a class token (embedClassB or embedClassA) at position 0; plus learned positions (embedPositionTwin); '
        'Transformer encoder (post-norm, GELU, d=128, feed-forward 512, no padding mask). '
        'Feature = encoder output at the class token (128). transformerTwinB has a final LayerNorm, '
        'transformerTwinA has none. '
        'msg: class token embedClassM, positions embedPositionMsg, transformerMsg with a final LayerNorm. '
        'head: classifier = LayerNorm(384) [.0] -> Linear(384,384) [.2] -> LayerNorm(384) [.3] -> Linear(384,2) [.4]. '
        'The last 5 columns of embedTwin are not used.')

    def __init__(self, sd, nhead):
        super().__init__()
        d = sd['embedTwin.weight'].shape[1]
        ff = sd['transformerTwinB.layers.0.linear1.weight'].shape[0]
        n_layers = len({k.split('.')[2] for k in sd if k.startswith('transformerTwinB.layers.')})
        self.embedTwin = nn.Embedding(*sd['embedTwin.weight'].shape)
        self.embedPositionTwin = nn.Parameter(torch.zeros(1, TWIN_MAX_LEN, d))
        self.embedPositionMsg = nn.Parameter(torch.zeros(1, MSG_MAX_LEN, d))
        self.embedMsg = nn.Embedding(*sd['embedMsg.weight'].shape)
        self.embedClassB = nn.Parameter(torch.zeros(1, 1, d))
        self.embedClassA = nn.Parameter(torch.zeros(1, 1, d))
        self.embedClassM = nn.Parameter(torch.zeros(1, 1, d))

        def encoder(final_norm):
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=ff,
                                               dropout=0.2, activation='gelu', batch_first=True)
            norm = nn.LayerNorm(d) if final_norm else None
            return nn.TransformerEncoder(layer, num_layers=n_layers, norm=norm,
                                         enable_nested_tensor=False, mask_check=False)
        self.transformerTwinB = encoder(True)
        self.transformerTwinA = encoder(False)
        self.transformerMsg = encoder(True)
        c = 3 * d
        self.classifier = nn.Sequential(nn.LayerNorm(c), nn.Dropout(p=0.5), nn.Linear(c, c),
                                        nn.LayerNorm(c), nn.Linear(c, 2))
        self.nhead = nhead
        self.load_state_dict(sd)

    @staticmethod
    def stage_of(name):
        if name.startswith(('embedTwin', 'embedPositionTwin')):
            return ('twin_b', 'twin_a')
        if name.startswith(('embedClassB', 'transformerTwinB')):
            return ('twin_b',)
        if name.startswith(('embedClassA', 'transformerTwinA')):
            return ('twin_a',)
        if name.startswith(('embedMsg', 'embedPositionMsg', 'embedClassM', 'transformerMsg')):
            return ('msg',)
        return ('head',)

    def last_norm(self, stage):
        """Name of the LayerNorm that writes the branch feature (the last operation of the branch)."""
        enc = {'twin_b': 'transformerTwinB', 'twin_a': 'transformerTwinA', 'msg': 'transformerMsg'}[stage]
        module = getattr(self, enc)
        if module.norm is not None:
            return enc + '.norm'
        return '%s.layers.%d.norm2' % (enc, len(module.layers) - 1)

    def _twin(self, tok, typ, cls, encoder):
        tok = tok[:, :TWIN_MAX_LEN - 1].long()
        typ = typ[:, :TWIN_MAX_LEN - 1]
        # The 5 token-type features replace the last 5 embedding dimensions.
        x = torch.cat((self.embedTwin(tok)[:, :, :-5], _one_hot_types(typ)), dim=2)
        x = torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1)
        return encoder(self.embedPositionTwin + x)[:, 0]

    def run_stage(self, stage, batch, acts):
        if stage == 'twin_b':
            return self._twin(batch['tok_b'], batch['typ_b'], self.embedClassB, self.transformerTwinB)
        if stage == 'twin_a':
            return self._twin(batch['tok_a'], batch['typ_a'], self.embedClassA, self.transformerTwinA)
        if stage == 'msg':
            x = self.embedMsg(batch['msg'][:, :MSG_MAX_LEN - 1].long())
            x = torch.cat((self.embedClassM.expand(x.shape[0], -1, -1), x), dim=1)
            return self.transformerMsg(self.embedPositionMsg + x)[:, 0]
        if stage == 'head':
            return self.classifier(torch.cat((acts['twin_b'], acts['twin_a'], acts['msg']), dim=1))
        raise KeyError(stage)


class TwinComb3(nn.Module):
    """Hybrid model ParCombCtl (Model14): a dense head over the features of three frozen models."""

    STAGES = ('twin_b', 'twin_a', 'msg', 'head')
    HEAD_LINEARS = ('combined_fc.0', 'combined_fc.3')
    OUT_BIAS = 'combined_fc.3.bias'
    SUBMODELS = (('cnn', 2), ('trans', 5), ('lstm', 0))       # prefix, archive model number
    TABLES = {'msg': ('cnn.embedMsg.weight', 'trans.embedMsg.weight', 'lstm.embedMsg.weight'),
              'code': ('cnn.embedTwin.weight', 'trans.embedTwin.weight', 'lstm.embedTwin.weight')}
    # Linear layers that read the branch features: (layer, first column, first feature unit, number of units).
    READERS = {'twin_b': (('cnn.fcB', 0, 0, 300), ('combined_fc.0', 450, 300, 128), ('lstm.fc1', 0, 428, 64)),
               'twin_a': (('cnn.fcA', 0, 0, 300), ('combined_fc.0', 578, 300, 128), ('lstm.fc1', 64, 428, 64)),
               'msg': (('cnn.fcM', 0, 0, 300), ('combined_fc.0', 706, 300, 128), ('lstm.fc3', 64, 428, 64))}
    DESCRIPTION = (
        'Hybrid of three archive models. Parameters have the prefixes cnn. (Model2), trans. (Model5) and '
        'lstm. (Model0). Each branch stage returns the branch features of the three models side by side: '
        'CNN pooled features (300), Transformer class-token feature (128), LSTM final states (64). '
        'head: CNN part = cat(cnn.fcB, cnn.fcA, cnn.fcM outputs) (450); LSTM part = lstm.fc3 output (64); '
        'Transformer part = the three class-token features (384). Order of the 898 inputs: CNN (0-449), Transformer (450-833: before, after, message), LSTM (834-897). '
        'combined_fc = Linear(898,898) [.0] -> ReLU -> Dropout -> Linear(898,2) [.3]. '
        'The sub-model layers cnn.fc, cnn.fc2, lstm.fc4 and trans.classifier are not used.')

    def __init__(self, sd, cnn, trans, lstm):
        super().__init__()
        self.cnn, self.trans, self.lstm = cnn, trans, lstm
        n = sd['combined_fc.0.weight'].shape[0]
        self.combined_fc = nn.Sequential(nn.Linear(n, n), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(n, 2))
        self.combined_fc.load_state_dict({k.split('.', 1)[1]: v for k, v in sd.items()})

    @staticmethod
    def stage_of(name):
        if name.startswith('combined_fc'):
            return ('head',)
        prefix, rest = name.split('.', 1)
        if prefix == 'cnn':
            return () if rest.startswith(('fc.', 'fc2.', 'norm')) else TwinCNN.stage_of(rest)
        if prefix == 'lstm':
            return () if rest.startswith(('fc4', 'fc2')) else TwinRNN.stage_of(rest)
        return () if rest.startswith('classifier') else TwinTransformer.stage_of(rest)

    def run_stage(self, stage, batch, acts):
        if stage != 'head':
            return torch.cat((self.cnn.run_stage(stage, batch, None), self.trans.run_stage(stage, batch, None),
                              self.lstm.run_stage(stage, batch, None)), dim=1)
        fb, fa, fm = acts['twin_b'], acts['twin_a'], acts['msg']
        f_cnn = torch.cat((self.cnn.fcB(fb[:, :300]), self.cnn.fcA(fa[:, :300]), self.cnn.fcM(fm[:, :300])), 1)
        f_tf = torch.cat((fb[:, 300:428], fa[:, 300:428], fm[:, 300:428]), 1)
        f = self.lstm.fc1(torch.cat((fb[:, 428:], fa[:, 428:]), 1))
        f_lstm = self.lstm.fc3(torch.cat((f, fm[:, 428:]), 1))
        return self.combined_fc(torch.cat((f_cnn, f_tf, f_lstm), 1))

    @classmethod
    def load(cls, archive_dir, model_no):
        sd = load_state_dict(checkpoint_path(archive_dir, model_no))
        subs = {prefix: build_model(archive_dir, no) for prefix, no in cls.SUBMODELS}
        return cls(sd, **subs)

    def export(self, archive_dir, model_no, out_dir):
        """Write the edited head, and each sub-model whose tensors differ from the archive file."""
        files = []
        path = os.path.join(out_dir, 'edited_' + os.path.basename(checkpoint_path(archive_dir, model_no)))
        torch.save(collections.OrderedDict(('combined_fc.' + k, v.clone()) for k, v in self.combined_fc.state_dict().items()), path)
        files.append(os.path.basename(path))
        for prefix, no in self.SUBMODELS:
            sd0 = load_state_dict(checkpoint_path(archive_dir, no))
            cur = getattr(self, prefix).state_dict()
            if any(not torch.equal(cur[k], sd0[k]) for k in sd0):
                for k in sd0:
                    sd0[k] = cur[k].clone()
                path = os.path.join(out_dir, 'edited_sub_Model%d_%s' % (no, REGISTRY[no][0]))
                torch.save(sd0, path)
                files.append(os.path.basename(path))
        return files


class CodeBertSC(nn.Module):
    """CodeBERT sequence classifier (Model15, checkpoint of the Hugging Face Trainer).

    One input: the token ids of the text "COMMIT: <message lines> CODE: <diff hunks>" (512 sub-word tokens).
    The engine sees the encoder as the message branch. The two code branches are constant placeholders.
    """

    STAGES = ('twin_b', 'twin_a', 'msg', 'head')
    HEAD_LINEARS = ('classifier.dense', 'classifier.out_proj')
    OUT_BIAS = 'classifier.out_proj.bias'
    WINDOW = (1, 512)
    BATCH = 8
    TABLES = {'msg': ('roberta.embeddings.word_embeddings.weight',), 'code': ()}
    READERS = {'msg': (('classifier.dense', 0, 0, 768),)}
    DESCRIPTION = (
        'RoBERTa-base encoder (12 layers, d=768) fine-tuned from CodeBERT. One input text: '
        '"COMMIT: <list of raw message lines> CODE: <list of raw diff hunks>", cut to 512 sub-word tokens. '
        'In this tool the text is the msg input. The stages twin_b and twin_a are constant placeholders. '
        'msg: encoder output at the first token (768). '
        'head: classifier.dense (768->768) -> tanh -> classifier.out_proj (768->2). '
        'The loss had no class weights. All embedding rows come from pre-training.')

    def __init__(self, sd):
        super().__init__()
        import transformers
        cfg = transformers.RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                                         intermediate_size=3072, max_position_embeddings=514, type_vocab_size=1,
                                         layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2, num_labels=2)
        net = transformers.RobertaForSequenceClassification(cfg)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if 'position_ids' not in k]
        if bad:
            raise RuntimeError('CodeBERT checkpoint keys do not match: %s' % bad[:6])
        self.roberta = net.roberta
        self.classifier = net.classifier

    @staticmethod
    def stage_of(name):
        return ('head',) if name.startswith('classifier') else ('msg',)

    def run_stage(self, stage, batch, acts):
        if stage in ('twin_b', 'twin_a'):
            return torch.zeros(batch[STAGE_KEY[stage]].shape[0], 1)
        if stage == 'msg':
            ids = batch['msg'].long()
            return self.roberta(input_ids=ids, attention_mask=(ids != 1).long())[0][:, 0, :]
        if stage == 'head':
            x = torch.tanh(self.classifier.dense(acts['msg']))
            return self.classifier.out_proj(x)
        raise KeyError(stage)

    @classmethod
    def load(cls, archive_dir, model_no):
        from safetensors.torch import load_file
        return cls(load_file(checkpoint_path(archive_dir, model_no)))

    def export(self, archive_dir, model_no, out_dir):
        from safetensors.torch import load_file, save_file
        sd = load_file(checkpoint_path(archive_dir, model_no))
        cur = dict(self.state_dict())
        for k in sd:
            sd[k] = cur[k].clone().contiguous()
        save_file(sd, os.path.join(out_dir, 'edited_model.safetensors'))
        return ['edited_model.safetensors']


STAGE_KEY = {'twin_b': 'tok_b', 'twin_a': 'tok_a'}

# Archive model number -> (file name, class, keyword arguments, label of the first article).
REGISTRY = {
    0: ('model_TwinRNN.pth', TwinRNN, {}, 'LSTM baseline (PatchRNN)'),
    2: ('model_TwinCNN.pth', TwinCNN, {}, 'Conv2D-E'),
    3: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 1}, 'Trans-1H'),
    4: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 2}, 'Trans-2H'),
    5: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 8}, 'Trans-8H'),
    6: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 8}, 'Trans-8H+B'),
    7: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 8}, 'Trans-8H+R'),
    8: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 16}, 'Trans-16H'),
    12: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 8}, 'Trans-8H-2E'),
    13: ('model_TwinTransformer.pth', TwinTransformer, {'nhead': 1}, 'Trans-1H-2E'),
    14: ('model_TwinComb3.pth', TwinComb3, {}, 'ParCombCtl (hybrid)'),
    15: ('results/checkpoint-4760/model.safetensors', CodeBertSC, {}, 'CodeBERT-SC'),
}


def checkpoint_path(archive_dir, model_no):
    name = REGISTRY[model_no][0]
    sub = 'content' if '/' in name else 'content/temp'
    return '%s/Model%d/%s/%s' % (archive_dir, model_no, sub, name)


def build_model(archive_dir, model_no):
    """Load one archive model. The model is in evaluation mode."""
    _, cls, kwargs, _ = REGISTRY[model_no]
    if hasattr(cls, 'load'):
        model = cls.load(archive_dir, model_no)
    else:
        model = cls(load_state_dict(checkpoint_path(archive_dir, model_no)), **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
