# Landing-Page Copywriting Fine-Tune

Fine-tunes a small open chat model to turn a short product brief into complete landing-page copy, emitted as one tagged line per element (`<h1>`, `<h2>`, `<h3>`, `<h4>`, `<p>`, `<button>`) so headline levels, body copy, and CTA buttons are all explicit and machine-parseable.

## Hardware

Trained and evaluated on a single 4-core Intel Sapphire Rapids box (AMX-BF16, 15 GB RAM, no GPU). Every throughput/memory number below was measured on this exact machine, CPU-only, bf16 autocast — no CUDA anywhere in this project.

## Model selection

The first choice, per the initial request, was **Gemma 4** (`google/gemma-4-E2B-it`, Apache-2.0, released 2026) — it does not fit this box:

- Text-only, it is 4.63B params (2.35B of that is the per-layer-embedding table, which `transformers` materializes in full) — 9.26 GB in bf16.
- A single 512-token forward+backward pass took 410 s and the process was OOM-killed at 10.9 GB anon RSS.
- Gemma 4 E4B is larger still (~8B). Qwen3.5-0.8B's GatedDeltaNet layers have no optimized CPU kernel, so it was ruled out without benchmarking.

Measured LoRA training throughput on this box (r=32, all-linear, bf16 autocast, gradient checkpointing on):

| Model | Throughput | Peak RSS |
|---|---|---|
| `LiquidAI/LFM2.5-350M` | 168 tok/s | 2.6 GB |
| `LFM2.5-1.2B-Instruct` | 69 tok/s | — |
| `Qwen3-0.6B` | 66 tok/s | — |
| `google/gemma-4-E2B-it` | ~1.25 tok/s, then OOM | OOM |

**Chosen base: [`LiquidAI/LFM2.5-350M`](https://huggingface.co/LiquidAI/LFM2.5-350M)** (instruct-tuned, released Aug 2026), pinned to revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`. Licensed under Liquid AI's [LFM Open License v1.0](https://www.liquid.ai/lfm-open-license) (license id `lfm1.0`) — check its commercial-use terms before shipping anything built on this adapter.

Counter-intuitive finding: gradient checkpointing **ON** is ~4x *faster* here, not slower. On a 4-core box, uncheckpointed activation memory thrashes the allocator; trading recomputation for a smaller memory footprint wins despite the extra forward pass (see the `scripts/train.py` module docstring).

## Data

**Sources** (full notes and per-source counts in `data/urls_gallery_sources.md`):
- **YC-OSS public API** (`yc-oss.github.io/api`, a community mirror of YC's 6,209-company directory) — website + one-liner + long description + industry + tags become the brief.
- **1,000 curated gallery homepages** — Landingfolio, SaaS Landing Page, One Page Love, two `awesome-saas` GitHub lists, and Indie Hackers — brief built from page title + meta description.

**Pipeline** (`scrape_pages.py` → `build_dataset.py` → `curate_dataset.py`):

| Stage | YC | Gallery | Total |
|---|---|---|---|
| Scraped | 4,817 | 992 | 5,809 |
| Passed build filters | 2,325 | 305 | 2,630 |
| Curated train / val / test | 1,175 / 85 / 106 | 127 / 15 / 14 | **1,302 / 100 / 120** |

Splits are by registrable domain — no domain appears in more than one split.

**`scrape_pages.py` extraction rules**: walks `h1`–`h4`, `p`, `button`, `a` under `<body>`, after stripping `script/style/nav/footer/iframe/...` and any element whose id/class matches a cookie/consent/nav/footer/sr-only regex. An `<a>` becomes a `<button>` only if it looks like a CTA (`role="button"`, or a class/role matching `btn|button|cta`). Element text prefers a non-empty `aria-label` over `get_text()`, then collapses immediately-repeated phrase blocks (a common animated-headline duplicate-DOM artifact). Dropped: `<p>` outside 2–90 words or that looks like an embedded SDK code line; `<button>` outside 1–7 words or matching a junk-phrase list (login/menu/cookie/close/next/…); symbol-only text; exact-text repeats within the same page.

**`build_dataset.py` filters** (drop reason, count): `no_h1` 1,167; `too_few_h2` 527 (<2); `too_few_button` 504 (<1); `non_english` 426; `too_few_p` 205 (<3); `dup_domain` 167; `h1_wordcount` 128 (outside 2–20 words); plus small buckets for forbidden phrases (`lorem ipsum`, `404`, `access denied`), short gallery descriptions, too-few-words, and a bad `<title>`. Surviving targets are capped at 420 words / 60 elements.

**`curate_dataset.py` rules**: parse every target line as `<tag>text</tag>` (drop on failure); per-line filters drop a `<p>` line that starts lowercase or with a non-letter/digit/quote/open-paren character, and drop a heading/button line that langdetect is confident (>0.9) is non-English (gated on ≥4 words plus a non-ASCII letter or ≥2 Romance/Germanic/Dutch function words, so plain English copy never reaches langdetect) — a record that loses >5 lines or >25% of its lines this way is dropped outright; the single `<h1>` must fall within the first 3 elements (else dropped, or promoted to position 0); then a quality battery: h1 2–16 words, p count 4–35, h2 count ≥2 and ≤45% of elements, longest same-tag run ≤8, 120–420 total words, an h3/h4 present or ≥2 buttons, no line >90 words, no button text repeated >2x, <4 pure-numeric lines, none of `cookie`/`javascript`/`subscribe to our newsletter`. Full drop histogram in `data/stats_curated.md`.

**Brief format** (real record `yc-00959`, from `data/train.jsonl`):
```
Product: Sophys
One-liner: AI Agents for Healthcare Intake
Industry: Healthcare
Tags: Artificial Intelligence, Analytics, Healthcare
```

**Target format** (same record, first 8 of 39 lines):
```
<h1>AI Agents for</h1>
<p>Custom Multimodal AI Agents for your Business</p>
<button>Get Started</button>
<h2>Wellness-Focused AI Solutions</h2>
<p>Purpose-built multimodal AI agents designed specifically for wellness platforms, mental health apps, and coaching services to enhance user engagement and deliver personalized support experiences.</p>
<h3>Empathetic User Interactions</h3>
<p>AI agents trained specifically for wellness communication, providing compassionate, supportive responses that maintain authentic connections and build trust with users.</p>
<h2>AI Companions for Every Wellness Journey</h2>
... (31 more lines)
```

Each example is wrapped as a 3-turn `messages` list — `system` (`scripts/prompt.py`'s `SYSTEM_PROMPT`), `user` (the brief), `assistant` (the target) — the format TRL's `SFTTrainer` expects.

## Training recipe

`scripts/train.py` runs TRL's `SFTTrainer` over the `messages` format with `assistant_only_loss=True`, using LFM2.5's own chat template — it ships `{% generation %}` markers, so the model's own template is used directly (TRL's `lfm2_training.jinja` is only a fallback for a template without them). `verify_loss_mask` (on by default) confirmed the unmasked span is exactly the assistant text plus `<|im_end|>`.

| Hyperparameter | Value | CLI flag |
|---|---|---|
| LoRA | r=32, alpha=64, dropout=0.05, all linear layers | `--lora-r 32 --lora-alpha 64 --lora-dropout 0.05 --target-modules all-linear` |
| Frozen | `embed_tokens`, `lm_head` | hardcoded (`modules_to_save=None`) |
| LR / schedule | 2e-4, cosine, 3% warmup | `--lr 2e-4 --lr-scheduler cosine --warmup-ratio 0.03` |
| Weight decay / grad clip | 0 / 1.0 | `--weight-decay 0 --max-grad-norm 1.0` |
| Optimizer | AdamW | `--optim adamw_torch` |
| Effective batch | 8 (1 × 8 accum) | `--per-device-bs 1 --grad-accum 8` |
| Max length | 1,408 tokens (keeps all examples; p50 = 927) | `--max-length 1408 --long-policy drop` |
| Epochs / steps | 3 epochs = 489 steps (1,302 ÷ 8 → 163 steps/epoch) | `--epochs 3` |
| Precision | bf16 autocast, fp32 master weights | default |
| Packing | off | hardcoded in `SFTConfig` |

1,302 training examples, ~1.16M tokens/epoch.

```bash
python3 scripts/train.py \
    --train data/train.jsonl --val data/val.jsonl \
    --out-dir runs/lfm2-lora-r32 \
    --epochs 3 --max-length 1408
```

Every run writes `run_config.json` into `--out-dir`, capturing the resolved base-model revision, full hyperparameters, dataset stats, the loss-mask check, package versions, and final metrics — the authoritative reproducibility record for that specific run.

**Recipe sources**: [Thinking Machines, "LoRA Without Regret"](https://thinkingmachines.ai/blog/lora/) (rank/alpha ratio, LoRA on all linear layers not just attention); [TRL `SFTTrainer` docs](https://huggingface.co/docs/trl/sft_trainer) and [TRL chat templates](https://huggingface.co/docs/trl/chat_templates) (`assistant_only_loss`, `{% generation %}` masking); [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora) (`LoraConfig`, `target_modules="all-linear"`); [Unsloth's LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) (LR/warmup/dropout ranges for small-model LoRA); [transformers CPU training guide](https://huggingface.co/docs/transformers/perf_train_cpu) (bf16 autocast + AMX, gradient-checkpointing tradeoffs); [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) and [HF Gemma 4 blog](https://huggingface.co/blog/gemma4) (param/memory accounting used to rule it out above).

## Evaluation

Harness detailed in `scripts/README_eval.md`; every script is model-agnostic (base or +LoRA) and CPU/bf16.

1. **`generate.py`** — batched CPU generation (left-padded), resumable, reports throughput.
2. **`evaluate.py`** — deterministic checks:
   - *Format validity*: each line matches `^<(h1|h2|h3|h4|p|button)>(.+)</\1>$`; exactly one `<h1>`, ≥2 `<h2>`, ≥3 `<p>`, ≥1 `<button>`, no duplicate lines, no stray text/markdown/code fences, length within 0.4–1.8x of the reference's word count.
   - *Structure*: mean tag/word counts vs. reference, Jensen–Shannon divergence between tag-proportion vectors, h1/button word-count distributions, imperative-CTA rate, first-element-is-h1 rate.
   - *Content*: ROUGE-1/2/L F1 (stemmed) — kept as a sanity check only, since n-gram overlap against one reference penalizes equally-good but differently-worded creative copy ([LLM-judge reporting practice](https://arxiv.org/pdf/2511.21140) argues for pairwise preference judging instead); brief-grounding — product-name mention rate, top-15 brief-keyword coverage, hallucinated-number count; distinct-2 and a 4-gram repetition flag.
3. **`eval_loss.py`** — teacher-forced mean per-token NLL/perplexity over assistant tokens only, base vs. fine-tuned.
4. **`make_judge_pack.py`** / **`score_judge.py`** — blind, position-randomized pairwise packs for an LLM judge (Claude Sonnet/Opus), scored on a 5-criteria + overall-preference rubric; reports per-system means, win/tie/loss, win rate excluding ties with a 95% bootstrap CI, and position bias.

### Results

<!-- RESULTS: fill -->

| Metric | Base | Fine-tuned |
|---|---|---|
| Format validity (all checks pass) | | |
| Exactly one `<h1>` | | |
| Perplexity (assistant tokens, held-out) | | |
| ROUGE-L F1 | | |
| Brief keyword coverage | | |
| Judge win rate | | |

<!-- RESULTS: fill -->

## Usage

```bash
pip install -r requirements.txt

# 1. Scrape
python3 scripts/scrape_pages.py --urls-json yc_all.json --source yc --out data/raw/raw_yc.jsonl
python3 scripts/scrape_pages.py --urls-txt data/urls_gallery.txt --source gallery --out data/raw/raw_gallery.jsonl

# 2. Build (brief, target) pairs, apply the quality filter, split by domain
python3 scripts/build_dataset.py --inputs data/raw/raw_yc.jsonl data/raw/raw_gallery.jsonl --out-dir data/dataset

# 3. Curate: per-line + record-level rules, hero normalization, re-split
python3 scripts/curate_dataset.py --input-dir data/dataset --output-dir data

# 4. Train (LoRA SFT)
python3 scripts/train.py --train data/train.jsonl --val data/val.jsonl \
    --out-dir runs/lfm2-lora-r32 --epochs 3 --max-length 1408

# 5. Generate predictions, base and fine-tuned, for comparison
python3 scripts/generate.py --model LiquidAI/LFM2.5-350M \
    --data data/test.jsonl --out runs/preds_base.jsonl
python3 scripts/generate.py --model LiquidAI/LFM2.5-350M --adapter runs/lfm2-lora-r32/adapter \
    --data data/test.jsonl --out runs/preds_ft.jsonl

# 6. Evaluate
python3 scripts/evaluate.py --preds runs/preds_ft.jsonl --out runs/results.json
python3 scripts/eval_loss.py --model LiquidAI/LFM2.5-350M --adapter runs/lfm2-lora-r32/adapter --data data/test.jsonl

# 7. Blind pairwise judge, base vs. fine-tuned
python3 scripts/make_judge_pack.py --a runs/preds_base.jsonl --b runs/preds_ft.jsonl \
    --names base,finetuned --out-dir runs/judge/
# fill in runs/judge/scores_*.jsonl per pack_*.md's rubric, then:
python3 scripts/score_judge.py --judge-dir runs/judge/
```

**Load the adapter for inference** (PEFT on top of the base model):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "LiquidAI/LFM2.5-350M"
adapter_path = "runs/lfm2-lora-r32/adapter"

tokenizer = AutoTokenizer.from_pretrained(base_path)
model = AutoModelForCausalLM.from_pretrained(base_path, dtype="bfloat16")
model = PeftModel.from_pretrained(model, adapter_path)
```

**Merge the adapter into a standalone model**:
```bash
python3 scripts/merge_adapter.py --base LiquidAI/LFM2.5-350M \
    --adapter runs/lfm2-lora-r32/adapter --out runs/lfm2-lora-r32/merged --verify
```

## Reproducibility

- Seeds: `--seed 42` throughout (`train.py`, `build_dataset.py`, `curate_dataset.py`'s re-split); `make_judge_pack.py` uses `--seed 7`; `score_judge.py`'s bootstrap uses `--seed 12345`.
- Base model: `LiquidAI/LFM2.5-350M`, revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- Versions (pinned in `requirements.txt`): Python 3.11, `torch` 2.14 (CPU build), `transformers` 5.17.0, `peft` 0.20.0, `trl` 1.13.0, `datasets` 5.0.1.
- Each training run also self-documents in `<out-dir>/run_config.json` (resolved revision, hyperparameters, dataset stats, loss-mask check, versions, metrics) — treat that file, not this README, as the source of truth for any specific run.

## Limitations

- **CPU-only compute budget** drove every choice above (model size, LoRA vs. full fine-tune, gradient checkpointing); none of this was compared against a GPU-trained baseline.
- **Real-world scraped data is noisy**: brief/target pairs inherit whatever a company's marketing copy actually claims, including unverifiable specifics, so the model can learn to assert confident-sounding but ungrounded numbers or claims.
- **License**: the base model ships under Liquid AI's own LFM Open License v1.0, not a standard OSI license — check its commercial-use terms before any commercial use of the adapter or a merged model. This repository's own code is AGPL-3.0 (see `LICENSE`).
- **Small model**: 350M params trades away coherence and world knowledge that larger instruction-tuned models have; it was the largest candidate that trains at a usable throughput on this box (see Model selection).

## References

- Thinking Machines, [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [TRL `SFTTrainer` docs](https://huggingface.co/docs/trl/sft_trainer) · [TRL chat templates](https://huggingface.co/docs/trl/chat_templates)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- Unsloth, [LoRA hyperparameters guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [transformers CPU training guide](https://huggingface.co/docs/transformers/perf_train_cpu)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [HF Gemma 4 blog](https://huggingface.co/blog/gemma4)
- [LLM-judge reporting practice (arXiv 2511.21140)](https://arxiv.org/pdf/2511.21140)
- [LFM Open License v1.0](https://www.liquid.ai/lfm-open-license)
