#!/usr/bin/env bash
# End-to-end eval pipeline: base vs LoRA-finetuned generation, teacher-forced
# loss/perplexity, format/structure/content metrics, and a blind judge pack
# -- then a compact side-by-side summary table.
#
# Every step is skipped (with a logged reason) if its output file already
# exists, so re-running after a partial run, a crash, or just to pick up
# newly-added judge/scores_*.jsonl only does the work that's still missing.
#
# Usage:
#   scripts/run_evals.sh              # run everything (skips completed steps)
#   scripts/run_evals.sh --dry-run    # print the commands, run nothing
set -euo pipefail

# --- config ------------------------------------------------------------------
BASE="LiquidAI/LFM2.5-350M"
ADAPTER="runs/lfm2-lora-r32/adapter"
TEST="data/test.jsonl"
OUT="evals"
N_JUDGE=60
THREADS=4
BATCH=8
MAX_NEW=640
# ------------------------------------------------------------------------------

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[run_evals] ERROR: unknown argument: $arg (only --dry-run/--help accepted)" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python3"

# step DESCRIPTION SENTINEL_FILE cmd...
# Skips (and logs) if SENTINEL_FILE already exists and is non-empty. In
# --dry-run mode nothing is ever executed -- the skip-check still runs so
# the printed preview matches what a real run would actually do.
step() {
  local desc="$1" sentinel="$2"
  shift 2
  if [[ -n "$sentinel" && -s "$sentinel" ]]; then
    echo "[run_evals] SKIP  ${desc}  (found ${sentinel})" >&2
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s:\n' "$desc" >&2
    printf '[dry-run]  ' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    return 0
  fi
  echo "[run_evals] RUN   ${desc}" >&2
  "$@"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[run_evals] --dry-run: printing commands only, nothing will be executed or created" >&2
else
  mkdir -p "$OUT"
fi

# --- 1-2. generation: finetuned (base + adapter), then base alone ------------
step "generate (finetuned = base+adapter)" "$OUT/preds_finetuned.jsonl" \
  "$PY" "$SCRIPT_DIR/generate.py" \
    --model "$BASE" --adapter "$ADAPTER" --data "$TEST" \
    --out "$OUT/preds_finetuned.jsonl" \
    --max-new-tokens "$MAX_NEW" --batch-size "$BATCH" --threads "$THREADS" --greedy

step "generate (base)" "$OUT/preds_base.jsonl" \
  "$PY" "$SCRIPT_DIR/generate.py" \
    --model "$BASE" --data "$TEST" \
    --out "$OUT/preds_base.jsonl" \
    --max-new-tokens "$MAX_NEW" --batch-size "$BATCH" --threads "$THREADS" --greedy

# --- 3. teacher-forced loss / perplexity, both systems ------------------------
step "eval_loss (finetuned)" "$OUT/loss_finetuned.json" \
  "$PY" "$SCRIPT_DIR/eval_loss.py" \
    --model "$BASE" --adapter "$ADAPTER" --data "$TEST" \
    --out "$OUT/loss_finetuned.json" --threads "$THREADS"

step "eval_loss (base)" "$OUT/loss_base.json" \
  "$PY" "$SCRIPT_DIR/eval_loss.py" \
    --model "$BASE" --data "$TEST" \
    --out "$OUT/loss_base.json" --threads "$THREADS"

# --- 4. format / structure / content metrics, both systems -------------------
step "evaluate (finetuned)" "$OUT/metrics_finetuned.json" \
  "$PY" "$SCRIPT_DIR/evaluate.py" \
    --preds "$OUT/preds_finetuned.jsonl" --out "$OUT/metrics_finetuned.json"

step "evaluate (base)" "$OUT/metrics_base.json" \
  "$PY" "$SCRIPT_DIR/evaluate.py" \
    --preds "$OUT/preds_base.jsonl" --out "$OUT/metrics_base.json"

# --- 5. blind, position-randomized judge pack ---------------------------------
step "make_judge_pack" "$OUT/judge/key.jsonl" \
  "$PY" "$SCRIPT_DIR/make_judge_pack.py" \
    --a "$OUT/preds_base.jsonl" --b "$OUT/preds_finetuned.jsonl" \
    --names base,finetuned --n "$N_JUDGE" --out-dir "$OUT/judge"

# --- 6. compact side-by-side summary table (pure python3 + json; no model) ---
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] summary table:" >&2
  echo "[dry-run]   python3 - reading ${OUT}/metrics_{base,finetuned}.json and ${OUT}/loss_{base,finetuned}.json" >&2
else
  echo "[run_evals] RUN   summary table" >&2
  "$PY" - "$OUT" <<'PYEOF'
import json
import os
import sys

out_dir = sys.argv[1]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


metrics = {s: load(os.path.join(out_dir, f"metrics_{s}.json")) for s in ("base", "finetuned")}
losses = {s: load(os.path.join(out_dir, f"loss_{s}.json")) for s in ("base", "finetuned")}

# (label, format spec, extractor(metrics_json, loss_json) -> float|None)
rows = [
    ("format validity",      "1%", lambda m, l: m["summary"]["overall_valid_rate"]),
    ("exactly_one_h1",       "1%", lambda m, l: m["summary"]["exactly_one_h1_rate"]),
    ("perplexity",           "3f", lambda m, l: l["perplexity"] if l else None),
    ("rougeL",               "3f", lambda m, l: m["summary"]["rougeL_f1"]),
    ("keyword coverage",     "1%", lambda m, l: m["summary"]["keyword_coverage"]),
    ("product-name mention", "1%", lambda m, l: m["summary"]["product_mention_rate"]),
    ("mean gen. tokens",     "1f", lambda m, l: m["content"]["mean_generated_tokens"]),
]


def fmt(v, spec):
    if v is None:
        return "n/a"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


label_w = max(len(r[0]) for r in rows)
header = f"{'metric':<{label_w}}  {'base':>12}  {'finetuned':>12}"
print(header)
print("-" * len(header))
for name, spec, fn in rows:
    cells = []
    for sys_name in ("base", "finetuned"):
        m, l = metrics[sys_name], losses[sys_name]
        v = fn(m, l) if m is not None else None
        cells.append(fmt(v, spec))
    print(f"{name:<{label_w}}  {cells[0]:>12}  {cells[1]:>12}")
PYEOF
fi

echo "[run_evals] DONE -> $OUT/" >&2
