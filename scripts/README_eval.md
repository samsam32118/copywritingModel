# Evaluation Harness

Model-agnostic SFT eval for landing-page copywriting (LFM2.5-350M / Qwen3.5-0.8B / gemma-4-E2B-it, base or +LoRA). CPU-only, bf16.

## Pipeline
1. `generate.py --model M [--adapter DIR] --data test.jsonl --out preds.jsonl` -- batched CPU generation (bf16, left padding), resumable (skips ids already in `--out`), prints throughput.
2. `evaluate.py --preds preds.jsonl --out results.json` -- format/structure/content metrics, plus a `results.md` table.
3. `eval_loss.py --model M [--adapter DIR] --data test.jsonl` -- teacher-forced mean per-token NLL/perplexity over assistant tokens only.
4. `make_judge_pack.py --a predsA.jsonl --b predsB.jsonl --names A,B --out-dir judge/` -- blind, position-randomized human-eval pack.
5. `score_judge.py --judge-dir judge/` -- joins `judge/scores_*.jsonl` with the hidden key, writes `judge/summary.md`.
6. `run_evals.sh` -- runs steps 1-5 end-to-end for base vs LoRA-finetuned (idempotent: skips any step whose output already exists), then prints a compact side-by-side metrics table. `--dry-run` previews the commands without running them.

`eval_common.py` holds the shared chat-template/model-loading/tag-parsing helpers every script above imports.

## Chat formatting
Uses `tokenizer.apply_chat_template`; passes `enable_thinking=False` when a template accepts it (dropped via try/except TypeError otherwise); folds the system prompt into the first user turn if a template rejects the system role; strips `<think>...</think>` (incl. an unclosed one) from generations.

## Format validity (evaluate.py)
Per line: `^<(h1|h2|h3|h4|p|button)>(.+)</\1>$`. Checks: all_lines_valid, exactly_one_h1, h2>=2, p>=3, button>=1, no_duplicate_lines, no_extra_text (no markdown/code fences/stray text), length_ok (pred word count within 0.4x-1.8x of the reference's). `valid` = all pass; each check's pass rate is reported.

## Structure
Mean tag counts and mean words/tag (pred vs ref), Jensen-Shannon divergence between mean tag-proportion vectors, H1 word-count distribution (% within 4-12 words), button word-count (% within 1-5 words), % buttons starting with an imperative verb, first-element-is-h1 rate.

## Content
ROUGE-1/2/L F1 (stemmed) vs reference; product-name mention rate (from the brief's `Product:` line); brief keyword coverage (top-15 non-stopword brief words found in the prediction); distinct-2; repetition flag (any 4-gram repeated >=3x); hallucinated-number count (numeric tokens in the prediction absent from the brief); mean generated tokens. Word/number metrics run on tag *content* only, so tag names never contaminate them.

## Judge pipeline
`make_judge_pack.py` seeds an RNG (`--seed`) to sample `--n` shared ids. With `--both-orders` (default), each id yields two judging items sharing that id -- order "a" (random position 1/2 assignment) and order "b" (positions swapped) -- item_id-keyed as `<id>#a`/`<id>#b`, with the hidden system mapping in `key.jsonl`. The two orders of an id always land in different `pack_XX.md` files, up to 8 items per file (`--no-both-orders` gives the legacy one-item-per-id mode instead). Judges fill `scores_XX.jsonl` per `pack_XX.md`'s rubric (5 criteria 1-5 + overall preference + a short note), keyed by `item_id`; see `judge/README.md` (written alongside the pack) for judge-facing instructions. `score_judge.py` reports per-system mean criteria (over all items), raw per-order win/tie/loss, and the primary metric -- **consistent win rate**: over ids judged in both orders, a system wins only if preferred in both orders (any disagreement, or a double tie, counts as a tie), with a 95% bootstrap CI over ids -- plus position bias (share of items where position 1 is preferred) and judge-agreement (share of dual-order ids where both orders agree).
