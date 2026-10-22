# Running Slipstream through Kapitsa's benchmark harness

Kapitsa's cross-engine harness (Apache-2.0, `kapitsa/bench/harness/`, stdlib-only Python, macOS) is the fairest
local-inference comparison we have found: one engine at a time under a GPU lock, cold prompts from a frozen corpus with
a nonce, decode rate over the middle of the stream, cached-TTFT and agent-replay phases, explicit sampler settings,
thinking off and *checked*, contention and repetition-loop flags, full provenance in every result, and a generated
table. It drives any OpenAI-compatible `/v1/chat/completions` server, so Slipstream plugs in with a config file.

This directory holds that config. Nothing here is needed to run or build Slipstream.

## Use it

```bash
# 1. Kapitsa's harness, next to this checkout
git clone --depth 1 https://github.com/nace-ai/kapitsa.git ~/src/kapitsa

# 2. Drop this config in
cp dev/benchmarks/kapitsa-harness/slipstream.json ~/src/kapitsa/bench/harness/configs/

# 3. Tell it where things are
cd ~/src/kapitsa/bench/harness
cp local.env.example local.env
$EDITOR local.env          # BENCH_MODELS; SLIPSTREAM_BIN, SLIPSTREAM_FLASH_MODEL, FLASH_NEXT_GGUF as needed

# 4. Resolve everything without starting a server first
python3 compare_engines.py configs/slipstream.json configs/flash-next.json --dry-run

# 5. Measure (one engine at a time; the harness takes /tmp/kapitsa-gpu.lock itself)
python3 compare_engines.py configs/slipstream.json configs/flash-next.json --only 'flash-*'

# 6. Table
python3 compare_table.py

# Optional: the agent-harness phases (cold-start, 1K..max sweep, shared-prefix
# and restart cache behaviour, 300 s sustained decode with a thermal check)
python3 compare_engines.py configs/slipstream.json --agentic
```

`cached TTFT` is measured in the default `prefill` phase (three append-only follow-up turns on the same conversation);
`--agentic` adds the sweep, the shared-prefix/restart cache checks and the sustain phase.

## Prepare the model package once

Slipstream's first launch against a GGUF shard directory spends 5–7 minutes writing `<model-dir>/prepared/`. Do that
outside the benchmark, then point `SLIPSTREAM_FLASH_MODEL` at the `prepared/` directory (the served id becomes
`local/prepared`, which is what the config's `model` field names):

```bash
./slipstream serve --model ~/models/swift-qwen38-flash-next-v3 --port 8090    # let it prepare
# Ctrl-C once "/v1/models" answers, then:
export SLIPSTREAM_FLASH_MODEL=~/models/swift-qwen38-flash-next-v3/prepared
```

## What the config measures, and what it refuses to

`flash-slipstream` is Slipstream's own defaults; `flash-slipstream-plain` is the same server with drafting effectively
off, for the same-engine baseline; `flash-slipstream-llamacpp` is the README's llama.cpp baseline under the harness's
rules (optional). Kapitsa's own Flash-Next config is `configs/flash-next.json` in their repo — **read its `note`**: on
a 128 GB Mac Kapitsa holds 105.9 GB of weights resident, while Slipstream streams 95.5 GiB on 64 GB. Those two rows are
not a like-for-like comparison of streaming designs, and the harness's memory and cached-TTFT columns are where the
difference is visible. On a **64 GB** Mac both engines stream, and that is the interesting test.

Two things to watch on the first run:

- **Reference prompt lengths.** `--ref-prompt-tokens 4000:4133,16000:16286` are the corpus prompts through
  Qwen3.8-Flash-Next's tokenizer and chat template (the numbers Kapitsa's and llama.cpp's Flash-Next configs use). The
  harness fails a run whose cold prompt comes back more than 1% off, which is what should happen if Slipstream's GGUF
  chat template differs. It prints the count it measured; put that in `bench_args` for this config.
- **Cached TTFT.** Slipstream has an in-memory prefix cache only (no SSD spill yet, see `docs/kapitsa-review.md` §3.1),
  so its cached-TTFT phase measures live-slot reuse and a restart loses it. That is a true result, not a bug in the run —
  it is the number the SSD prefix cache would change.

## License

The harness lives in `nace-ai/kapitsa` (Apache-2.0). This config file is ours; keep the attribution note in the config's
`note` fields if you edit it.
