"""Command line interface.

  python -m scalpel.cli init  --run RUN --archive ARCHIVE --bundle BUNDLE --model N
  python -m scalpel.cli call  --run RUN TOOL ['{"json": "arguments"}']
  python -m scalpel.cli tools
  python -m scalpel.cli agent --run RUN [--llm MODEL_ID] [--max-turns N]

'call' lets a person or an external agent operate the same tools as the API loop.
"""
import argparse
import json
import sys
import warnings

warnings.filterwarnings('ignore')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='scalpel')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init')
    p.add_argument('--run', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--bundle', required=True)
    p.add_argument('--model', type=int, required=True)
    p.add_argument('--max-commits', type=int, default=None)
    p.add_argument('--max-evaluations', type=int, default=None)
    p.add_argument('--guard-metric', choices=['mcc', 'f1'], default=None)
    p = sub.add_parser('call')
    p.add_argument('--run', required=True)
    p.add_argument('--actor', default='operator')
    p.add_argument('tool')
    p.add_argument('args', nargs='?', default='{}')
    sub.add_parser('tools')
    p = sub.add_parser('agent')
    p.add_argument('--run', required=True)
    p.add_argument('--llm', default=None)
    p.add_argument('--max-turns', type=int, default=60)
    a = ap.parse_args(argv)

    if a.cmd == 'tools':
        from .tools import TOOLS
        for t in TOOLS:
            print('%-20s %s' % (t['name'], t['description']))
        return 0

    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    from .engine import Session

    if a.cmd == 'init':
        limits = {}
        if a.max_commits:
            limits['max_commits'] = a.max_commits
        if a.max_evaluations:
            limits['max_evaluations'] = a.max_evaluations
        if a.guard_metric:
            limits['guard_metric'] = a.guard_metric
        s = Session.create(a.run, a.archive, a.bundle, a.model, limits=limits)
        s.save()
        from .instruments import metrics
        print(json.dumps({'run': a.run, 'model': s.label, 'diag': metrics(s, 'diag')['baseline'],
                          'guard': metrics(s, 'guard')['baseline']}))
        return 0
    if a.cmd == 'call':
        from .tools import dispatch
        s = Session(a.run)
        try:
            args = json.loads(a.args)
        except json.JSONDecodeError as exc:
            print(json.dumps({'error': 'The arguments are not valid JSON: %s' % exc}))
            return 2
        print(json.dumps(dispatch(s, a.tool, args, actor=a.actor), ensure_ascii=False))
        return 0
    if a.cmd == 'agent':
        from .agent import run_agent
        run_agent(a.run, llm=a.llm, max_turns=a.max_turns)
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())
