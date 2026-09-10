#!/usr/bin/env python3
"""Compute mean per-token NLL and perplexity over ASSISTANT tokens only.

Everything up to and including the generation prompt (system + user turn,
plus whatever role-opening tokens the chat template inserts for the
assistant turn) is masked out of the loss; only the target/assistant tokens
the model was trained to produce are scored. The assistant span is located
by tokenizing the prompt-only chat template (add_generation_prompt=True)
and taking the suffix of the full conversation's tokenization, per the
project's convention (see eval_common.assistant_token_span).

Runs one example at a time (bf16, no_grad) -- simple and exact, no padding
edge cases to reason about; CPU-appropriate for eval-set sizes.

Example:
    python scripts/eval_loss.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --data data/dataset/test.jsonl --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import eval_common as ec


def parse_args():
    p = argparse.ArgumentParser(
        description="Mean per-token NLL / perplexity over assistant tokens only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Base model name or local path (AutoModelForCausalLM).")
    p.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter directory to apply on top of --model.")
    p.add_argument("--data", required=True, help="Path to a JSONL eval set (id, brief, target, messages).")
    p.add_argument("--n", type=int, default=None, help="Only use the first N records of --data (default: all).")
    p.add_argument("--out", default="eval_loss.json", help="Output JSON path for the aggregate metrics.")
    p.add_argument("--threads", type=int, default=4, help="torch intra-op CPU thread count.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    import torch.nn.functional as F

    print(f"[eval_loss] loading model={args.model} adapter={args.adapter} threads={args.threads} dtype=bfloat16 (cpu)", file=sys.stderr)
    t_load0 = time.time()
    model, tokenizer = ec.load_model_and_tokenizer(args.model, adapter=args.adapter, dtype=torch.bfloat16, threads=args.threads)
    print(f"[eval_loss] model loaded in {time.time() - t_load0:.1f}s", file=sys.stderr)

    data = ec.read_jsonl(args.data, n=args.n)
    print(f"[eval_loss] scoring {len(data)} records", file=sys.stderr)

    total_nll_sum = 0.0  # sum of per-token NLL across the whole corpus (token-weighted)
    total_tokens = 0
    per_example_ppl = []
    skipped = 0

    t0 = time.time()
    with torch.no_grad():
        for r in data:
            full_ids, prompt_len = ec.assistant_token_span(tokenizer, r, enable_thinking=False)
            if prompt_len >= len(full_ids):
                # Degenerate record (empty target after tokenization): nothing to score.
                skipped += 1
                continue

            input_ids = torch.tensor([full_ids], dtype=torch.long)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0].float()  # [seq, vocab], fp32 for a stable cross-entropy

            # logits[t-1] predicts token t, so to score assistant tokens
            # full_ids[prompt_len:] we need logits[prompt_len-1 : -1].
            pred_logits = logits[prompt_len - 1 : -1, :]
            target_ids = input_ids[0, prompt_len:]

            nll = F.cross_entropy(pred_logits, target_ids, reduction="sum")
            n_assistant_tokens = target_ids.numel()

            total_nll_sum += nll.item()
            total_tokens += n_assistant_tokens
            per_example_ppl.append(float(torch.exp(nll / n_assistant_tokens)))

    elapsed = time.time() - t0
    n_scored = len(data) - skipped

    if total_tokens == 0:
        print("[eval_loss] ERROR: no assistant tokens found across the dataset", file=sys.stderr)
        sys.exit(1)

    mean_nll = total_nll_sum / total_tokens  # corpus-level, token-weighted
    perplexity = float(torch.exp(torch.tensor(mean_nll)))
    mean_of_per_example_ppl = sum(per_example_ppl) / len(per_example_ppl)

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "n_examples": n_scored,
        "n_skipped_empty_target": skipped,
        "n_assistant_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "mean_of_per_example_perplexity": mean_of_per_example_ppl,
        "seconds": elapsed,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"[eval_loss] DONE: n={n_scored} assistant_tokens={total_tokens} "
        f"mean_nll={mean_nll:.4f} perplexity={perplexity:.3f} "
        f"({elapsed:.1f}s, {n_scored / max(elapsed, 1e-9):.3f} ex/s) -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
