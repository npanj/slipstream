#!/bin/zsh
# One benchmark arm: raw generate-sample capture + totals + token dump.
# Usage: run_arm.sh <prefix>   (model: prepared swift flash-next v3, default cache)
ROOT=/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream
OUT=/tmp/prof
PREFIX=$1
cd $ROOT
# Pre-flight: need plenty of headroom (30.5 GiB expert cache + Metal working
# set + file cache). Refuse rather than swap.
FREE_GIB=$(vm_stat | awk '/Pages free:/{f=$3} /Pages inactive:/{i=$3} /Pages speculative:/{s=$3} END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s); print int((f+i+s)*16384/1073741824)}')
# Default-cache runs proved safe 10+ times; the guard floor (10 GiB)
# plus steady swap are the real tripwires. 34 GiB-cache arms stay risky.
if [ "$FREE_GIB" -lt 28 ]; then echo "REFUSING $PREFIX: only ${FREE_GIB}GiB available"; exit 2; fi
if pgrep -x generate-sample >/dev/null || pgrep -x slipstream >/dev/null; then echo "REFUSING $PREFIX: another engine is running"; exit 2; fi
PROMPTS_JSON=$ROOT/build/qwen4exp-speed-prompts.json
JOINED=$(.venv/bin/python -c "import json;d=json.load(open('$PROMPTS_JSON'));print(';'.join(','.join(map(str,d[k])) for k in d))")
MODEL=$HOME/models/swift-qwen38-flash-next-v3/prepared
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
