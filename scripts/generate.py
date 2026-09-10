#!/usr/bin/env python3
"""Generate landing-page copy predictions from a (optionally PEFT-adapted)
causal LM over a JSONL evaluation set, for later scoring by evaluate.py.

Model-agnostic: works with any AutoModelForCausalLM + AutoTokenizer whose
tokenizer ships a chat template (e.g. LiquidAI/LFM2.5-350M, Qwen/Qwen3.5-0.8B,
google/gemma-4-E2B-it), with or without a PEFT LoRA adapter directory.

Example:
    python scripts/generate.py --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --data data/dataset/test.jsonl --out preds.jsonl --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import eval_common as ec


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate landing-page copy predictions for a JSONL eval set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Base model name or local path (AutoModelForCausalLM).")
    p.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter directory to apply on top of --model.")
    p.add_argument("--data", required=True, help="Path to a JSONL eval set (id, brief, target, messages).")
    p.add_argument("--n", type=int, default=None, help="Only use the first N records of --data (default: all).")
    p.add_argument("--out", default="preds.jsonl", help="Output JSONL of predictions.")
    p.add_argument("--max-new-tokens", type=int, default=640, help="Max new tokens to generate per example.")
    p.add_argument("--batch-size", type=int, default=4, help="Generation batch size (uses left padding).")
    p.add_argument("--greedy", action="store_true", help="Force greedy decoding (default when neither --temperature nor --top-p is given).")
    p.add_argument("--temperature", type=float, default=None, help="Sampling temperature (implies sampling unless --greedy is also set).")
    p.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top-p (implies sampling unless --greedy is also set).")
    p.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty for generation.")
    p.add_argument("--threads", type=int, default=4, help="torch intra-op CPU thread count.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch

    do_sample = (args.temperature is not None or args.top_p is not None) and not args.greedy

    print(f"[generate] loading model={args.model} adapter={args.adapter} threads={args.threads} dtype=bfloat16 (cpu)", file=sys.stderr)
    t_load0 = time.time()
    model, tokenizer = ec.load_model_and_tokenizer(args.model, adapter=args.adapter, dtype=torch.bfloat16, threads=args.threads)
    tokenizer.padding_side = "left"  # required for correct batched generation
    eos_ids = ec.eos_token_ids(tokenizer, model)
    print(f"[generate] model loaded in {time.time() - t_load0:.1f}s, eos_ids={sorted(eos_ids)}", file=sys.stderr)

    data = ec.read_jsonl(args.data, n=args.n)
    done_ids = ec.existing_ids(args.out)
    todo = [r for r in data if r["id"] not in done_ids]
    print(f"[generate] {len(data)} records selected, {len(done_ids)} already in {args.out}, {len(todo)} to run", file=sys.stderr)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        if args.temperature is not None:
            gen_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
    print(f"[generate] decoding: do_sample={do_sample} {gen_kwargs}", file=sys.stderr)

    out_f = open(args.out, "a", encoding="utf-8")
    total_new_tokens = 0
    total_gen_seconds = 0.0
    n_done = 0

    with torch.no_grad():
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            prompts = [
                ec.apply_chat_template_safe(
                    tokenizer,
                    ec.get_messages(r, include_assistant=False),
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                for r in batch
            ]
            enc = ec.encode_prompts(tokenizer, prompts)
            input_len = enc["input_ids"].shape[1]

            t0 = time.time()
            output_ids = model.generate(**enc, **gen_kwargs)
            batch_seconds = time.time() - t0

            for i, r in enumerate(batch):
                gen_ids = output_ids[i, input_len:].tolist()
                if eos_ids:
                    cut = next((j for j, tid in enumerate(gen_ids) if tid in eos_ids), len(gen_ids))
                    gen_ids = gen_ids[:cut]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                text = ec.strip_think(text)

                rec = {
                    "id": r["id"],
                    "brief": r["brief"],
                    "reference": r.get("target", ""),
                    "prediction": text,
                    "gen_seconds": batch_seconds / len(batch),
                    "n_new_tokens": len(gen_ids),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_new_tokens += len(gen_ids)

            total_gen_seconds += batch_seconds
            n_done += len(batch)
            out_f.flush()
            print(
                f"[generate] {n_done}/{len(todo)} done "
                f"(running throughput {n_done / total_gen_seconds:.3f} seq/s) "
                f"batch={batch_seconds:.1f}s",
                file=sys.stderr,
            )

    out_f.close()

    if n_done > 0 and total_gen_seconds > 0:
        seq_per_s = n_done / total_gen_seconds
        tok_per_s = total_new_tokens / total_gen_seconds
        print(
            f"[generate] DONE: {n_done} sequences, {total_new_tokens} new tokens, "
            f"{total_gen_seconds:.1f}s generation time -> "
            f"{seq_per_s:.3f} seq/s, {tok_per_s:.2f} tok/s throughput",
            file=sys.stderr,
        )
    else:
        print(f"[generate] DONE: nothing new to generate (all {len(data)} ids already in {args.out})", file=sys.stderr)


if __name__ == "__main__":
    main()
