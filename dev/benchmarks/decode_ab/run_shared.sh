#!/bin/zsh
# Shared guesser comparison: identical TEXT prompts, per-model tokenization.
# Usage: run_shared.sh <model-dir> <prefix>   (24 GiB cache for safety)
ROOT=/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream
OUT=/tmp/prof
MODEL=$1; PREFIX=$2
cd $ROOT
FREE_GIB=$(vm_stat | awk '/Pages free:/{f=$3} /Pages inactive:/{i=$3} /Pages speculative:/{s=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s); print int((f+i+s)*16384/1073741824)}')
if [ "$FREE_GIB" -lt 28 ]; then echo "REFUSING $PREFIX: only ${FREE_GIB}GiB available"; exit 2; fi
if pgrep -x generate-sample >/dev/null || pgrep -x slipstream >/dev/null; then echo "REFUSING $PREFIX: another engine is running"; exit 2; fi
JOINED=$(.venv/bin/python -c "
from transformers import AutoTokenizer
import json, os
tok = AutoTokenizer.from_pretrained(os.path.join('$MODEL', 'tokenizer'))
code = open('$ROOT/runtime/ops/GDN.cpp').read()[:6000]
texts = {
 'echo': 'Reproduce the following code EXACTLY, character for character, with no commentary:\n\n' + code,
 'math': 'Janet\'s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market? Think it through step by step.',
 'prose': 'Explain how a transformer\'s attention mechanism works to a software engineer who knows linear algebra but not machine learning.',
}
out = {}
for n, t in texts.items():
    r = tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True)
    out[n] = tok.encode(r, add_special_tokens=False)
print(';'.join(','.join(map(str, out[k])) for k in out))
" 2>/dev/null)
$ROOT/dev/benchmarks/guarded.py --floor-gib 10 --max-seconds 2400 -- \
  $ROOT/build/engine-tests/generate-sample $ROOT/build/splash.metallib "$MODEL" 256 "$JOINED" \
  > $OUT/${PREFIX}.raw.json 2> $OUT/${PREFIX}.raw.log
echo "exit=$? arm=$PREFIX"
.venv/bin/python - "$PREFIX" <<'EOF'
import json,sys
from collections import Counter
p=sys.argv[1]
rows=[json.loads(l) for l in open(f'/tmp/prof/{p}.raw.json') if l.startswith('{')]
if not rows: print(f'{p}: NO ROWS (see /tmp/prof/{p}.raw.log tail)'); raise SystemExit(3)
names=['echo','math','prose']
for n,r in zip(names,rows):
    acc=r.get('step_accepted',[]); c=Counter(acc); tot=sum(acc); s=len(acc)
    toks=len(r['generated_tokens']); ms=sum(r['step_ms'])
    hist=' '.join(f'{k}:{c[k]}' for k in sorted(c))
    print(f'{p}/{n}: accept/step={tot/max(s,1):.2f} {1000*toks/ms:.1f} tok/s hist[{hist}]')
json.dump([r['generated_tokens'] for r in rows], open(f'/tmp/prof/{p}.toks.json','w'))
EOF
