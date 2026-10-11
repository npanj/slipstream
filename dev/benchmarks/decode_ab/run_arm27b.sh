#!/bin/zsh
# 27B arm: raw generate-sample capture + totals + token dump.
# Usage: run_arm27b.sh <prefix>   (model: swift-qwen38-27b-splash-hq)
ROOT=/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream
OUT=/tmp/prof
PREFIX=$1
cd $ROOT
FREE_GIB=$(vm_stat | awk '/Pages free:/{f=$3} /Pages inactive:/{i=$3} /Pages speculative:/{s=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s); print int((f+i+s)*16384/1073741824)}')
# 27B pins far less than Flash-Next (~16 GB model, small dense cache).
# Swap is the real tripwire: refuse if it is already in use.
SWAP_USED=$(sysctl -n vm.swapusage | awk '{print $7}')
if [ "$FREE_GIB" -lt 18 ]; then echo "REFUSING $PREFIX: only ${FREE_GIB}GiB available"; exit 2; fi
echo "swap in use before run: $SWAP_USED"
if pgrep -x generate-sample >/dev/null || pgrep -x slipstream >/dev/null; then echo "REFUSING $PREFIX: another engine is running"; exit 2; fi
MODEL=$HOME/models/swift-qwen38-27b-splash-hq
JOINED=$(.venv/bin/python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL/tokenizer', local_files_only=True)
texts = {
 'palindrome': 'Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.',
 'ducks': 'Janet\'s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers\' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market? Think it through.',
 'explain': 'Explain how a transformer\'s attention mechanism works to a software engineer who knows linear algebra but not machine learning.',
 'lru': 'Implement an LRU cache class in Python with get and put in O(1), then write unit tests for it.',
}
out = {}
for n, t in texts.items():
    r = tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True)
    out[n] = tok.encode(r, add_special_tokens=False)
import json
json.dump(out, open('/tmp/prof/prompts27b.json', 'w'))
print(';'.join(','.join(map(str, out[k])) for k in out))
")
$ROOT/dev/benchmarks/guarded.py --floor-gib 10 --max-seconds 2400 -- \
  $ROOT/build/engine-tests/generate-sample $ROOT/build/splash.metallib "$MODEL" 128 "$JOINED" \
  > $OUT/${PREFIX}.raw.json 2> $OUT/${PREFIX}.raw.log
echo "exit=$? arm=$PREFIX"
.venv/bin/python - "$PREFIX" <<'EOF'
import json,sys
p=sys.argv[1]
rows=[json.loads(l) for l in open(f'/tmp/prof/{p}.raw.json') if l.startswith('{')]
if not rows: print(f'{p}: NO ROWS (see /tmp/prof/{p}.raw.log tail)'); raise SystemExit(3)
t=sum(len(r['generated_tokens']) for r in rows); ms=sum(sum(r['step_ms']) for r in rows); s=sum(r['decode_steps'] for r in rows)
print(f'{p}: {1000*t/ms:.1f} tok/s ({t} tok, {t/s:.2f} tok/step, {ms/s:.1f} ms/step)')
json.dump([r['generated_tokens'] for r in rows], open(f'/tmp/prof/{p}.toks.json','w'))
EOF
