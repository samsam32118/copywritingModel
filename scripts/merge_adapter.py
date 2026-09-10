#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and save a standalone bf16 model.

The merge itself is done in fp32 (W + BA is computed at full precision) and the
result is cast to bf16 only when saving, which keeps the merged weights as close
as possible to what the adapter produced at training time.

Example:
    python3 scripts/merge_adapter.py \
        --base LiquidAI/LFM2.5-350M \
        --adapter runs/lfm2-lora-r32/adapter \
        --out runs/lfm2-lora-r32/merged
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into its base model and save a standalone model + tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base", required=True, help="Base model name or path (must match what the adapter was trained on).")
    p.add_argument("--adapter", required=True, help="PEFT adapter directory (adapter_config.json + adapter_model.safetensors).")
    p.add_argument("--out", required=True, help="Output directory for the merged model + tokenizer.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Saved weight dtype.")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (torch intra-op + OMP_NUM_THREADS).")
    p.add_argument("--verify", action="store_true", help="Run one forward pass through both the PEFT model and the merged model and report max logit delta.")
    return p.parse_args(argv)


def log(msg: str) -> None:
    print(f"[merge] {msg}", file=sys.stderr, flush=True)


def rss_gb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except OSError:
        return float("nan")


def main(argv=None):
    args = parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    adapter_cfg_path = os.path.join(args.adapter, "adapter_config.json")
    adapter_cfg = {}
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        trained_on = adapter_cfg.get("base_model_name_or_path")
        if trained_on and trained_on != args.base:
            log(f"WARNING: adapter was trained on {trained_on!r}, merging into {args.base!r}")
    else:
        log(f"WARNING: no adapter_config.json in {args.adapter}")

    t0 = time.time()
    log(f"loading base model {args.base} in fp32 (merge precision)")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    log(f"base loaded in {time.time() - t0:.1f}s, rss={rss_gb():.2f} GB")

    log(f"applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, dtype=torch.float32)
    model.eval()

    ref_logits = None
    sample_ids = None
    tokenizer = None
    # Prefer the tokenizer saved next to the adapter (it carries the exact chat
    # template used for training); fall back to the base model's.
    tok_src = args.adapter if os.path.isfile(os.path.join(args.adapter, "tokenizer_config.json")) else args.base
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    log(f"tokenizer loaded from {tok_src}")

    if args.verify:
        sample_ids = tokenizer("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        with torch.no_grad():
            ref_logits = model(**sample_ids).logits.float().clone()

    t1 = time.time()
    log("merging LoRA weights into the base (fp32)")
    merged = model.merge_and_unload()
    log(f"merged in {time.time() - t1:.1f}s, rss={rss_gb():.2f} GB")

    if args.verify and ref_logits is not None:
        with torch.no_grad():
            new_logits = merged(**sample_ids).logits.float()
        delta = (new_logits - ref_logits).abs().max().item()
        log(f"verify: max |logit delta| between PEFT-wrapped and merged model = {delta:.3e} (fp32 merge)")

    merged = merged.to(save_dtype)
    dtype_str = str(save_dtype).replace("torch.", "")
    # transformers 5.x renamed config.torch_dtype -> config.dtype
    if hasattr(merged.config, "dtype"):
        merged.config.dtype = dtype_str
    else:  # pragma: no cover - transformers 4.x
        merged.config.torch_dtype = dtype_str

    os.makedirs(args.out, exist_ok=True)
    log(f"saving merged model ({args.dtype}) to {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)

    meta = {
        "base": args.base,
        "adapter": os.path.abspath(args.adapter),
        "adapter_config": adapter_cfg,
        "save_dtype": args.dtype,
        "merge_dtype": "fp32",
        "tokenizer_source": tok_src,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    files = sorted(os.listdir(args.out))
    log(f"done in {time.time() - t0:.1f}s | peak rss ~{rss_gb():.2f} GB | files: {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
