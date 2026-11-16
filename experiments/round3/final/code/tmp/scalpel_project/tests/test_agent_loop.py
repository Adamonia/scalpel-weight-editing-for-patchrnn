"""Test of the API agent loop with a scripted client (no network, no API key).

Run: python tests/test_agent_loop.py ARCHIVE BUNDLE
The scripted "model" edits only an unused tensor, thus the holdout result equals the baseline.
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scalpel.agent import run_agent
from scalpel.engine import Session

ARCH, BUNDLE = sys.argv[1], sys.argv[2]


def tool(i, name, args):
    return NS(stop_reason='tool_use', content=[
        NS(type='text', text='step %d' % i),
        NS(type='tool_use', id='toolu_%d' % i, name=name, input=args)])


SCRIPT = [
    tool(1, 'overview', {}),
    tool(2, 'edit_tensor', {'name': 'fc2.weight', 'op': 'scale', 'value': 0.0}),
    tool(3, 'evaluate', {}),
    tool(4, 'commit', {'rationale': 'fc2 is not used by the forward pass', 'error_type': 'dead parameters'}),
    tool(5, 'no_such_tool', {}),
    NS(stop_reason='end_turn', content=[NS(type='text', text='I stop here.')]),      # The loop must nudge.
    tool(6, 'finalize', {'note': 'scripted test'}),
    NS(stop_reason='end_turn', content=[NS(type='text', text='Report: one neutral edit.')]),
]


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        # Each tool_use of the last assistant turn must have a tool_result in the next user turn.
        msgs = kw['messages']
        if len(msgs) >= 2 and isinstance(msgs[-1]['content'], list) and msgs[-2]['role'] == 'assistant':
            want = [b.id for b in msgs[-2]['content'] if getattr(b, 'type', '') == 'tool_use']
            got = [b['tool_use_id'] for b in msgs[-1]['content'] if b.get('type') == 'tool_result']
            assert want == got, (want, got)
            types = [b['type'] for b in msgs[-1]['content']]
            assert types == sorted(types, key=lambda t: t != 'tool_result'), 'tool_result blocks must be first'
        assert kw['tools'] and kw['system'] and kw['model'] == 'fake-model'
        self.calls.append(len(msgs))
        return SCRIPT[len(self.calls) - 1]


def main():
    d = tempfile.mkdtemp(prefix='scalpel_agent_')
    shutil.rmtree(d)
    Session.create(d, ARCH, BUNDLE, 0).save()
    client = NS(messages=FakeMessages())
    s = run_agent(d, llm='fake-model', max_turns=20, client=client, verbose=False)
    assert s.final is not None and s.final['n_commits'] == 1
    h = s.final['parts']['holdout']
    assert h['n_fixed'] == 0 and h['n_broken'] == 0
    assert len(client.messages.calls) == len(SCRIPT)
    lines = open(os.path.join(d, 'transcript.jsonl'), encoding='utf-8').read().strip().split('\n')
    assert len(lines) >= 2 * len(SCRIPT) - 2
    shutil.rmtree(d)
    print('AGENT LOOP TEST PASSED (%d scripted turns)' % len(SCRIPT))


if __name__ == '__main__':
    main()
