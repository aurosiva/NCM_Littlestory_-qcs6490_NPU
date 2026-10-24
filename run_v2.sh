#!/usr/bin/env bash
# Waits for the baseline generator run to finish, exports it, then trains and exports v2.
# Logs: logs/queue.log, logs/tokenizer_v2.log, logs/generator_v2.log
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
export TMPDIR="$PWD/.smoke/tmp"   # root partition is nearly full; keep temp files on /home
mkdir -p "$TMPDIR" logs
say() { echo "[$(date '+%F %T')] $*" | tee -a logs/queue.log; }

say "waiting for baseline generator to finish"
while pgrep -f "python train_generator.py$" >/dev/null || pgrep -f "python train_generator.py " >/dev/null; do sleep 30; done
grep -q "^saved ckpt/generator.pt" logs/generator.log && say "baseline finished" || say "WARNING: baseline log has no final save line"

say "exporting baseline -> edge/"
$PY export_edge.py --ckpt ckpt --out edge --int8 > logs/export_base.log 2>&1
say "baseline export: $(grep parity logs/export_base.log)"

say "training v2 tokenizer -> ckpt_v2/"
if ! $PY train_tokenizer.py --preset v2 --out ckpt_v2 > logs/tokenizer_v2.log 2>&1; then
    if grep -q "OutOfMemory" logs/tokenizer_v2.log; then
        say "v2 tokenizer OOM at batch 256; retrying with batch 128"
        $PY train_tokenizer.py --preset v2 --out ckpt_v2 --batch 128 > logs/tokenizer_v2.log 2>&1 || { say "v2 tokenizer FAILED"; exit 1; }
    else
        say "v2 tokenizer FAILED (see logs/tokenizer_v2.log)"; exit 1
    fi
fi
say "v2 tokenizer done: $(grep eval logs/tokenizer_v2.log | tail -1)"

say "training v2 generator (2 x 64 = effective batch 128, same as baseline)"
$PY train_generator.py --preset v2 --out ckpt_v2 --batch 64 --accum 2 > logs/generator_v2.log 2>&1 \
    || { say "v2 generator FAILED (see logs/generator_v2.log)"; exit 1; }

say "exporting v2 -> edge_v2/"
$PY export_edge.py --ckpt ckpt_v2 --out edge_v2 --int8 > logs/export_v2.log 2>&1
say "v2 export: $(grep parity logs/export_v2.log)"
say "all done"
