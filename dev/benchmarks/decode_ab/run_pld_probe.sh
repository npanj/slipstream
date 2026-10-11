#!/bin/zsh
# Repetition-heavy probe for PLD: prompt contains code, answer must echo it.
# Usage: run_pld_probe.sh <prefix>   (24 GiB cache for machine safety)
ROOT=/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream
OUT=/tmp/prof
PREFIX=$1
cd $ROOT
FREE_GIB=$(vm_stat | awk '/Pages free:/{f=$3} /Pages inactive:/{i=$3} /Pages speculative:/{s=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s); print int((f+i+s)*16384/1073741824)}')
if [ "$FREE_GIB" -lt 28 ]; then echo "REFUSING $PREFIX: only ${FREE_GIB}GiB available"; exit 2; fi
if pgrep -x generate-sample >/dev/null || pgrep -x slipstream >/dev/null; then echo "REFUSING $PREFIX: another engine is running"; exit 2; fi
MODEL=$HOME/models/swift-qwen38-flash-next-v3/prepared
JOINED=$(.venv/bin/python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL/tokenizer', local_files_only=True)
code = open('$ROOT/runtime/ops/GDN.cpp').read()[:6000]
texts = {
 'echo_code': 'Reproduce the following code EXACTLY, character for character, with no commentary:\n\n' + code,
 'extend_code': 'Here is a Python module:\n\n' + open('$ROOT/server/agent/tools.py').read()[:4000] + '\n\nAdd type hints to every function signature above and print the FULL updated file, keeping everything else identical.',
}
out = {}
for n, t in texts.items():
    r = tok.apply_chat_template([{'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True)
    out[n] = tok.encode(r, add_special_tokens=False)
import json
print(';'.join(','.join(map(str, out[k])) for k in out))
")
$ROOT/dev/benchmarks/guarded.py --floor-gib 10 --max-seconds 2400 -- \
  $ROOT/build/engine-tests/generate-sample $ROOT/build/splash.metallib "$MODEL" 256 "$JOINED" \
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
