"""Tool registry. The agent loop and the command line use the same dispatcher."""
import json

from . import edits, instruments, report
from .engine import ScalpelError

_WHERE = {'type': 'object', 'description': (
    'Sample filter. Keys (all optional): msg_len [lo,hi], len_b [lo,hi], len_a [lo,hi], code_len [lo,hi], '
    'untr_msg_min, untr_code_min, label (0/1), pred (0/1), msg_has [tokens], msg_regex, msg_lacks_regex, '
    'code_has [tokens], code_regex, ids [sample ids].')}
_PART = {'type': 'string', 'enum': ['diag', 'guard'], 'description': 'Data part. Default diag.'}
_TABLE = {'type': 'string', 'enum': ['msg', 'code'], 'description': 'Embedding table: msg (embedMsg) or code (embedTwin).'}


def _spec(name, fn, desc, props=None, required=(), mutates=False):
    return {'name': name, 'fn': fn, 'mutates': mutates, 'description': desc,
            'input_schema': {'type': 'object', 'properties': props or {}, 'required': list(required)}}


def _evaluate(s):
    return s.evaluate()


def _commit(s, rationale, error_type):
    rec = s.commit(rationale, error_type)
    return {'commit': rec['commit'], 'error_type': error_type,
            'diag': {k: rec['diag'][k] for k in ('before', 'after', 'n_fixed', 'n_broken')},
            'guard': {k: rec['guard'][k] for k in ('before', 'after', 'n_fixed', 'n_broken')}}


def _rollback(s):
    return {'pending_edits_removed': s.rollback()}


def _finalize(s, note=''):
    res = report.finalize(s, note=note)
    return {'holdout': res['parts']['holdout'], 'weight_diff': res['weight_diff'],
            'edited_checkpoint': res['edited_checkpoint'], 'n_commits': res['n_commits']}


TOOLS = [
    _spec('overview', instruments.overview,
          'Model structure, tensor names and shapes, the stage that each tensor feeds, split sizes, current metrics.'),
    _spec('metrics', instruments.metrics, 'Confusion counts and metrics of one part (current, committed, baseline).',
          {'part': _PART}),
    _spec('error_census', instruments.error_census,
          'Error rates by message length, code length, empty before-version and untrained-token counts.', {'part': _PART}),
    _spec('slice_metrics', instruments.slice_metrics,
          'Metrics of the samples that satisfy a filter, compared with the other samples.',
          {'where': _WHERE, 'part': _PART}, ['where']),
    _spec('list_errors', instruments.list_errors,
          'Misclassified diag samples with margin, lengths and a message preview.',
          {'kind': {'type': 'string', 'enum': ['FN', 'FP', 'all']},
           'sort': {'type': 'string', 'enum': ['confident', 'borderline']},
           'limit': {'type': 'integer'}, 'offset': {'type': 'integer'}, 'where': _WHERE}),
    _spec('show_sample', instruments.show_sample,
          'Raw commit message, raw diff, model input tokens, prediction and branch contributions of one diag sample.',
          {'id': {'type': 'integer'}, 'max_msg_chars': {'type': 'integer'}, 'max_diff_chars': {'type': 'integer'},
           'code_tokens': {'type': 'integer'}}, ['id']),
    _spec('branch_attribution', instruments.branch_attribution,
          'Mean contribution of the before-code, after-code and message branch to the margin, for TP, TN, FP, FN.',
          {'part': _PART, 'where': _WHERE}),
    _spec('token_attribution', instruments.token_attribution,
          'Occlusion attribution for one diag sample: margin change when one input position becomes padding.',
          {'id': {'type': 'integer'}, 'branch': {'type': 'string', 'enum': ['msg', 'twin_b', 'twin_a']},
           'top': {'type': 'integer'}}, ['id']),
    _spec('vocab_search', instruments.vocab_search,
          'Vocabulary rows that match a regex, with training document counts for each class.',
          {'table': _TABLE, 'regex': {'type': 'string'}, 'limit': {'type': 'integer'},
           'only_untrained': {'type': 'boolean'}, 'fullmatch': {'type': 'boolean'}}, ['table', 'regex']),
    _spec('untrained_summary', instruments.untrained_summary,
          'Embedding rows that training never changed: counts, shape classes, examples, error rates.', {'table': _TABLE}),
    _spec('token_effect', instruments.token_effect,
          'In-context effect of tokens: mean margin change when all occurrences of the token become padding.',
          {'table': _TABLE, 'tokens': {'type': 'array', 'items': {'type': 'string'}}, 'regex': {'type': 'string'},
           'part': _PART, 'max_tokens': {'type': 'integer'}}, ['table']),
    _spec('effect_by_frequency', instruments.effect_by_frequency,
          'Occlusion effect of single token positions grouped by the training count of the token: size of the effect '
          'and its mean help for the correct class. Finds token groups that add noise.',
          {'table': _TABLE, 'per_bucket': {'type': 'integer'}, 'part': _PART}),
    _spec('solo_probe', instruments.solo_probe,
          'Margin of a message with only one token, minus the margin of an empty message.',
          {'tokens': {'type': 'array', 'items': {'type': 'string'}}, 'regex': {'type': 'string'},
           'limit': {'type': 'integer'}}),
    _spec('row_neighbors', instruments.row_neighbors, 'Nearest embedding rows of a token by cosine similarity.',
          {'table': _TABLE, 'token': {'type': 'string'}, 'k': {'type': 'integer'},
           'trained_only': {'type': 'boolean'}}, ['table', 'token']),
    _spec('unit_report', instruments.unit_report,
          'First-order help of each feature unit of one branch for the correct class; the most harmful units.',
          {'stage': {'type': 'string', 'enum': ['twin_b', 'twin_a', 'msg']}, 'top': {'type': 'integer'}, 'part': _PART},
          ['stage']),
    _spec('bias_scan', instruments.bias_scan,
          'Metrics of the diag part for shifts of the output bias (decision threshold). No weight change.',
          {'deltas': {'type': 'array', 'items': {'type': 'number'}}}),
    _spec('weight_read', instruments.weight_read, 'Statistics of one tensor and the values of a small slice.',
          {'name': {'type': 'string'}, 'index': {'type': 'array'}, 'max_values': {'type': 'integer'}}, ['name']),
    _spec('edit_embedding_rows', edits.edit_embedding_rows,
          'EDIT. Write a recipe vector into embedding rows. targets: {"tokens":[...]} or {"regex":..., '
          '"only_untrained":true, "max_train_count":k}. recipe: {"type":"blend","sources":[{"token":t,"weight":w}]} | '
          '{"type":"mean_of","regex":...,"min_train_count":3,"label":"pos"|"neg"} | {"type":"mean_trained"} | '
          '{"type":"zero"} | {"type":"pad"} | {"type":"scale","factor":f} | '
          '{"type":"compose","split":regex,"min_train_count":5} (split the token text, average the rows of its trained '
          'parts) | {"type":"nearest_numeric","k":1} (prefix+number tokens: copy the trained row with the nearest '
          'number). new_row = (1-mix)*old + mix*vector.',
          {'table': _TABLE, 'targets': {'type': 'object'}, 'recipe': {'type': 'object'}, 'mix': {'type': 'number'}},
          ['table', 'targets', 'recipe'], mutates=True),
    _spec('edit_tensor', edits.edit_tensor,
          'EDIT. Direct write into a tensor slice. op: set, add or scale. index: one entry for each dimension: '
          'int, list of ints, {"range":[a,b]} or null.',
          {'name': {'type': 'string'}, 'op': {'type': 'string', 'enum': ['set', 'add', 'scale']},
           'value': {}, 'index': {'type': 'array'}}, ['name', 'op', 'value'], mutates=True),
    _spec('edit_feature_gain', edits.edit_feature_gain,
          'EDIT. Scale the effect of branch feature units on the head (the diag mean of each unit is kept). '
          'TwinRNN and TwinCNN only.',
          {'stage': {'type': 'string', 'enum': ['twin_b', 'twin_a', 'msg']},
           'units': {'type': 'array', 'items': {'type': 'integer'}}, 'factor': {'type': 'number'}},
          ['stage', 'units', 'factor'], mutates=True),
    _spec('edit_branch_gain', edits.edit_branch_gain,
          'EDIT. Scale the deviation of one branch feature vector from its diag mean. factor 0 removes the branch, '
          'factor 1 changes nothing. Exact for all model types.',
          {'stage': {'type': 'string', 'enum': ['twin_b', 'twin_a', 'msg']}, 'factor': {'type': 'number'}},
          ['stage', 'factor'], mutates=True),
    _spec('edit_recenter', edits.edit_recenter,
          'EDIT. Shift the output bias so that the mean margin of the diag part has a target value (default: the mean '
          'margin of the committed weights). Use it after an edit that moves all margins in one direction. No label is used.',
          {'target_mean': {'type': 'number'}}, mutates=True),
    _spec('edit_rank1', edits.edit_rank1,
          'EDIT. Closed-form rank-one edit of a head Linear layer. The mean layer input of the selected diag samples '
          'gets a margin shift of delta_margin; other inputs change little (ridge-regularized second moments).',
          {'layer': {'type': 'string'}, 'where': _WHERE, 'delta_margin': {'type': 'number'},
           'ridge': {'type': 'number'}}, ['layer', 'where', 'delta_margin'], mutates=True),
    _spec('evaluate', _evaluate,
          'Effect of the pending edits on diag and guard: metrics before and after, fixed and broken samples.',
          mutates=True),
    _spec('commit', _commit,
          'Accept the pending edits. Refused if the guard metric (MCC, or F1 if the session is set to F1) of diag or guard decreases. Give the hypothesis (rationale) '
          'and a short name of the error type that the edit corrects.',
          {'rationale': {'type': 'string'}, 'error_type': {'type': 'string'}}, ['rationale', 'error_type'], mutates=True),
    _spec('rollback', _rollback, 'Remove all pending edits.', mutates=True),
    _spec('finalize', _finalize,
          'End of the session: evaluate the committed weights on the sealed holdout part, write the edited '
          'checkpoint and lock the session. Call this one time only.',
          {'note': {'type': 'string'}}, mutates=True),
]
BY_NAME = {t['name']: t for t in TOOLS}


def api_tools():
    """Tool definitions for the Anthropic Messages API."""
    return [{'name': t['name'], 'description': t['description'], 'input_schema': t['input_schema']} for t in TOOLS]


def dispatch(session, name, args, actor='agent'):
    """Run one tool. The call and a short result are written to the journal."""
    if name not in BY_NAME:
        return {'error': 'Unknown tool %s. Tools: %s' % (name, sorted(BY_NAME))}
    tool = BY_NAME[name]
    try:
        result = tool['fn'](session, **(args or {}))
    except ScalpelError as exc:
        result = {'error': str(exc)}
    except TypeError as exc:
        result = {'error': 'Bad arguments for %s: %s' % (name, exc)}
    text = json.dumps(result, ensure_ascii=False)
    session.log('tool', {'actor': actor, 'tool': name, 'args': args or {},
                         'result': result if len(text) <= 6000 else {'truncated': text[:6000]}})
    if tool['mutates'] or name in ('metrics', 'error_census', 'slice_metrics', 'list_errors', 'branch_attribution',
                                   'unit_report', 'untrained_summary', 'token_effect', 'solo_probe', 'show_sample', 'effect_by_frequency',
                                   'token_attribution', 'overview', 'bias_scan'):
        session.save()
    return result
