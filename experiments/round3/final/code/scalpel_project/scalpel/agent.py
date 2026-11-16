"""Autonomous agent loop on the Anthropic Messages API.

The loop gives Claude the tools of scalpel.tools. Claude reads diagnostics,
writes weight edits, and ends the session with finalize.

The API key comes from the environment variable ANTHROPIC_API_KEY.
The model id comes from --llm or from SCALPEL_LLM (default: claude-opus-5-5).
"""
import json
import os
import time

from .engine import Session
from .tools import api_tools, dispatch

DEFAULT_LLM = 'claude-opus-5-5'

SYSTEM_PROMPT = """You are a weight-editing agent. You repair a trained neural network that identifies security patches.
The network reads a commit (code before the change, code after the change, commit message) and gives two logits.
margin = logit(security) - logit(non-security).

Your task: find the parts of the network that do the task incorrectly, and correct them with direct edits of the weights.
You do not train the network. You do not use gradient descent on the weights. You write weight values with the edit tools.

Data protocol (the tools enforce it):
- diag part: you can read each sample, its label and the model output. Use it to find error types and to design edits.
- guard part: you see only aggregate metrics. It tells you if an edit also works on samples that you did not read.
- holdout part: sealed. Only finalize reads it, one time. The reported result comes from this part.

Procedure:
1. Call overview and error_census. Read the architecture text. Note unused tensors and suspicious structure.
2. Make hypotheses about mechanisms that cause errors. Examples: embedding rows that training never changed,
   tokens with a wrong learned polarity, a branch that the head ignores, a shortcut on an input property,
   a decision bias. Test each hypothesis with the instruments (slice_metrics, branch_attribution, token_effect,
   token_attribution, unit_report, vocab_search, show_sample) before you edit.
3. Write one edit or a small group of edits for one hypothesis. Call evaluate.
4. Call commit with the hypothesis and a short error-type name if the edit helps. Call rollback if it does not.
   commit is refused if the MCC of the diag part or of the guard part decreases.
5. Repeat. Prefer a small number of edits with a clear mechanism. Do not tune many numbers to the diag part:
   the guard part and the holdout part will show that as overfitting.
6. Call finalize when no more justified edits are available or when the limits are near.
   Then write a short report: error types found, edits made, evidence, and what did not work.

State facts from tool results only. If a hypothesis fails, say so."""

KICKOFF = 'Start the session. The model checkpoint and the data are loaded. Begin with overview.'


def _dump(block):
    if hasattr(block, 'model_dump'):
        return block.model_dump()
    if isinstance(block, dict):
        return block
    return dict(vars(block))


def run_agent(run_dir, llm=None, max_turns=60, client=None, max_tokens=8000, verbose=True):
    """Run the loop until finalize, the end of the turn budget, or a stop by the model."""
    llm = llm or os.environ.get('SCALPEL_LLM', DEFAULT_LLM)
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    s = Session(run_dir)
    s.log('agent_start', {'llm': llm, 'max_turns': max_turns})
    transcript = os.path.join(run_dir, 'transcript.jsonl')
    messages = [{'role': 'user', 'content': KICKOFF}]
    nudges = 0

    def record(role, content):
        with open(transcript, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps({'t': round(time.time(), 2), 'role': role, 'content': content},
                                ensure_ascii=False, default=str) + '\n')

    record('user', KICKOFF)
    for turn in range(max_turns):
        resp = client.messages.create(model=llm, max_tokens=max_tokens, system=SYSTEM_PROMPT,
                                      tools=api_tools(), messages=messages)
        messages.append({'role': 'assistant', 'content': resp.content})
        record('assistant', [_dump(b) for b in resp.content])
        if verbose:
            for b in resp.content:
                if getattr(b, 'type', None) == 'text' and b.text.strip():
                    print('[agent] ' + b.text.strip())
        if resp.stop_reason == 'tool_use':
            results = []
            for b in resp.content:
                if getattr(b, 'type', None) != 'tool_use':
                    continue
                out = dispatch(s, b.name, dict(b.input or {}), actor=llm)
                if verbose:
                    print('[tool] %s %s' % (b.name, json.dumps(b.input, ensure_ascii=False)[:200]))
                results.append({'type': 'tool_result', 'tool_use_id': b.id,
                                'content': json.dumps(out, ensure_ascii=False), 'is_error': 'error' in out})
            left = max_turns - turn - 1
            if s.final is None and left in (5, 2):
                results.append({'type': 'text', 'text': 'Turn budget: %d turns left. Commit or roll back, '
                                                        'then call finalize.' % left})
            messages.append({'role': 'user', 'content': results})
            record('user', results)
            continue
        if resp.stop_reason == 'max_tokens':
            messages.append({'role': 'user', 'content': 'Continue.'})
            record('user', 'Continue.')
            continue
        # end_turn, refusal, or another stop reason.
        if s.final is None and resp.stop_reason == 'end_turn' and nudges < 2:
            nudges += 1
            text = 'The session is not final. Continue the work, or call finalize and then write the report.'
            messages.append({'role': 'user', 'content': text})
            record('user', text)
            continue
        break
    s.log('agent_end', {'llm': llm, 'final': s.final is not None, 'turns_used': turn + 1})
    return s
