import os
_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get('SCALPEL_RUNS', os.path.join(_HERE, '..', 'runs'))
BUNDLE = os.environ.get('SCALPEL_BUNDLE', os.path.join(_HERE, '..', 'bundle'))
ARCHIVE = os.environ.get('SCALPEL_ARCHIVE', os.path.join(_HERE, '..', '..', 'model-archive'))

import sys, json, warnings, re
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_HERE, '..'))
from scalpel.engine import Session
from scalpel.tools import dispatch
no = int(sys.argv[1]); ids = [int(x) for x in sys.argv[2].split(',')]
mc = int(sys.argv[3]) if len(sys.argv) > 3 else 260; dc = int(sys.argv[4]) if len(sys.argv) > 4 else 420
s = Session(os.path.join(RUNS, 'M%d' % no))
for i in ids:
    r = dispatch(s, 'show_sample', {'id': i, 'max_msg_chars': mc, 'max_diff_chars': dc, 'code_tokens': 5}, actor='claude-fable-5-2')
    if 'error' in r: print(i, r['error']); continue
    msg = re.sub(r'\s+', ' ', r['raw_message']).strip()
    diff = re.sub(r'[ \t]+', ' ', r['raw_diff']).strip()
    diff = re.sub(r'\n\s*\n', '\n', diff)
    print('### id %d label=%d margin=%.1f lens m%d b%d a%d hunks%d\nMSG: %s\nDIFF: %s' % (i, r['label'], r['margin'], r['lengths']['msg'], r['lengths']['before'], r['lengths']['after'], r['lengths']['hunks'], msg, diff))
s.save()
