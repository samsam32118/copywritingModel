#!/usr/bin/env python3
"""LoRA SFT for landing-page copywriting on CPU (LFM2.5-350M by default).

Trains with loss on the ASSISTANT turn only, using the chat template's
`{% generation %}` markers (LFM2.5's own template ships them, so training and
inference use the byte-identical template that scripts/eval_common.py applies
at generation time).

Tuned for a 4-core CPU box:
  * LoRA r=32 / alpha=64 on all linear projections (no embed_tokens/lm_head)
  * bf16 autocast (`bf16=True` + `use_cpu=True`) over fp32 master weights,
    which is what feeds the AMX-BF16 units on Sapphire Rapids
  * gradient checkpointing ON -- counter-intuitively ~4x FASTER here, because
    keeping activations out of the allocator avoids thrashing it

Example (smoke test):
    python3 scripts/train.py --train data/dataset/train.jsonl \
        --val data/dataset/val.jsonl --out-dir runs/smoke \
        --max-train-examples 16 --max-val-examples 8 --max-steps 3 --max-length 512

Example (full run):
    python3 scripts/train.py --train data/dataset/train.jsonl \
        --val data/dataset/val.jsonl --out-dir runs/lfm2-lora-r32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

# Thread env vars must be set before torch is imported, so argparse runs first
# and the heavyweight imports happen inside main().

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="LoRA supervised fine-tune of a chat LM on landing-page copy (CPU-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / data
    p.add_argument("--model", default="LiquidAI/LFM2.5-350M", help="Base model name or local path.")
    p.add_argument("--train", required=True, help="Training JSONL (id, brief, target, messages).")
    p.add_argument("--val", default=None, help="Validation JSONL. Omit to skip evaluation.")
    p.add_argument("--out-dir", required=True, help="Output directory (checkpoints, adapter, logs).")

    # Optimisation
    p.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate.")
    p.add_argument("--per-device-bs", type=int, default=1, help="Per-device train batch size.")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (effective batch = bs*accum).")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="Fraction of total steps used for LR warmup.")
    p.add_argument("--lr-scheduler", default="cosine", help="LR scheduler type (cosine, linear, constant, ...).")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max-norm.")
    p.add_argument("--optim", default="adamw_torch", help="Optimizer (HF `optim` name).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # LoRA
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (2x rank per LoRA-Without-Regret).")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    p.add_argument(
        "--target-modules",
        default="all-linear",
        help="'all-linear' or a comma-separated list of module suffixes. embed_tokens/lm_head are never trained.",
    )

    # Sequence handling
    p.add_argument("--max-length", type=int, default=1024, help="Max tokens per example (prompt + completion).")
    p.add_argument(
        "--long-policy",
        choices=["drop", "truncate"],
        default="drop",
        help="What to do with examples longer than --max-length. 'drop' protects the end-of-turn token from being cut.",
    )
    p.add_argument("--max-train-examples", type=int, default=None, help="Cap on training examples (after filtering).")
    p.add_argument("--max-val-examples", type=int, default=100, help="Cap on validation examples.")

    # Schedule control / IO
    p.add_argument("--max-steps", type=int, default=-1, help="Hard cap on optimizer steps (-1 = use --epochs). For smoke tests.")
    p.add_argument("--save-steps", type=int, default=100, help="Checkpoint every N optimizer steps.")
    p.add_argument("--save-total-limit", type=int, default=2, help="Max checkpoints kept on disk.")
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N optimizer steps.")
    p.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy.")
    p.add_argument("--eval-steps", type=int, default=None, help="Eval every N steps (only with --eval-strategy steps).")
    p.add_argument("--resume-from-checkpoint", default=None, help="Path to a checkpoint dir (or 'auto' for the latest in --out-dir).")

    # Runtime
    p.add_argument("--threads", type=int, default=4, help="CPU threads (torch intra-op + OMP_NUM_THREADS).")
    p.add_argument(
        "--chat-template",
        choices=["auto", "model", "trl-lfm2"],
        default="auto",
        help="auto: use the model's own template if it has {%% generation %%} markers, else TRL's lfm2_training.jinja.",
    )
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing (SLOWER on this CPU -- see module docstring).")
    p.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast (pure fp32 training).")
    p.add_argument("--skip-mask-check", action="store_true", help="Skip the assistant-only loss-mask verification printout.")
    p.add_argument("--dry-run", action="store_true", help="Prepare data/model/trainer and run the mask check, then exit without training.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[train] {msg}", file=sys.stderr, flush=True)


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def peak_rss_gb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return float("nan")


def rss_gb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except OSError:
        return float("nan")


def resolve_base_revision(model_id: str) -> dict:
    """Best-effort resolution of the exact base-model commit hash."""
    info = {"model": model_id, "revision": None, "revision_source": None, "local_path": None}
    if os.path.isdir(model_id):
        info["local_path"] = os.path.abspath(model_id)
        info["revision_source"] = "local_dir"
        return info
    try:
        from huggingface_hub import constants as hf_constants

        cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or hf_constants.HF_HUB_CACHE
    except Exception:
        cache = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache, "models--" + model_id.replace("/", "--"))
    ref = os.path.join(repo_dir, "refs", "main")
    if os.path.isfile(ref):
        with open(ref, "r", encoding="utf-8") as f:
            info["revision"] = f.read().strip()
        info["revision_source"] = "hf_cache_refs_main"
    elif os.path.isdir(os.path.join(repo_dir, "snapshots")):
        snaps = sorted(os.listdir(os.path.join(repo_dir, "snapshots")))
        if snaps:
            info["revision"] = snaps[-1]
            info["revision_source"] = "hf_cache_snapshot_dir"
    if os.path.isdir(repo_dir):
        info["local_path"] = repo_dir
    return info


def percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {}
    s = sorted(lengths)
    return {
        "n": len(s),
        "total_tokens": int(sum(s)),
        "min": s[0],
        "p50": percentile(s, 0.50),
        "mean": round(statistics.fmean(s), 1),
        "p90": percentile(s, 0.90),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def measure_and_filter(records, tokenizer, chat_template, max_length, long_policy, split_name):
    """Tokenize every record once to get (total, assistant) token counts, then
    drop or keep-for-truncation the over-length ones. Returns
    (kept_records, stats_dict)."""
    total_lens, asst_lens, over = [], [], []
    kept = []
    no_assistant = 0
    for rec in records:
        msgs = rec["messages"]
        out = tokenizer.apply_chat_template(
            msgs,
            chat_template=chat_template,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        n_tok = len(out["input_ids"])
        n_asst = int(sum(out.get("assistant_masks", []) or []))
        if n_asst == 0:
            no_assistant += 1
            continue
        total_lens.append(n_tok)
        asst_lens.append(n_asst)
        if n_tok > max_length:
            over.append(n_tok)
            if long_policy == "drop":
                continue
        kept.append(rec)

    stats = {
        "split": split_name,
        "records_in": len(records),
        "records_kept": len(kept),
        "records_over_max_length": len(over),
        "records_without_assistant_tokens": no_assistant,
        "long_policy": long_policy,
        "max_length": max_length,
        "over_length_token_counts": length_stats(over) if over else None,
        "total_token_stats": length_stats(total_lens),
        "assistant_token_stats": length_stats(asst_lens),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# JSONL training log
# ---------------------------------------------------------------------------


def make_jsonl_logger_callback(path, total_steps):
    from transformers import TrainerCallback

    class JsonlLogger(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.f = None
            self.last_tokens = 0
            self.last_time = None
            # Tokens already counted when training starts (non-zero on resume),
            # so throughput can be reported for THIS run only.
            self.tokens_at_begin = 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")
            self.t0 = time.time()
            self.last_time = self.t0
            self.last_tokens = float(getattr(state, "num_input_tokens_seen", 0) or 0)
            self.tokens_at_begin = self.last_tokens

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.f is None or not logs:
                return
            now = time.time()
            elapsed = now - self.t0
            tokens = float(getattr(state, "num_input_tokens_seen", 0) or 0)
            dt = max(now - self.last_time, 1e-9)
            interval_tps = (tokens - self.last_tokens) / dt
            overall_tps = tokens / elapsed if elapsed > 0 else 0.0
            step = int(state.global_step)
            steps_left = max(total_steps - step, 0)
            eta = (elapsed / step) * steps_left if step > 0 else None

            rec = {
                "step": step,
                "total_steps": total_steps,
                "epoch": round(float(state.epoch), 4) if state.epoch is not None else None,
                "elapsed_s": round(elapsed, 2),
                "tokens_seen": int(tokens),
                "tokens_per_sec": round(interval_tps, 2),
                "tokens_per_sec_overall": round(overall_tps, 2),
                "eta_s": round(eta, 1) if eta is not None else None,
                "rss_gb": round(rss_gb(), 3),
                "peak_rss_gb": round(peak_rss_gb(), 3),
            }
            for k, v in logs.items():
                if k in ("loss", "eval_loss", "grad_norm", "learning_rate", "mean_token_accuracy",
                         "eval_mean_token_accuracy", "entropy", "num_tokens", "train_loss", "eval_runtime"):
                    rec[k] = v
            rec.setdefault("lr", logs.get("learning_rate"))
            self.f.write(json.dumps(rec) + "\n")
            self.f.flush()
            self.last_tokens = tokens
            self.last_time = now

            if "loss" in logs or "eval_loss" in logs:
                bits = [f"step {step}/{total_steps}"]
                if "loss" in logs:
                    bits.append(f"loss {logs['loss']:.4f}")
                if "eval_loss" in logs:
                    bits.append(f"eval_loss {logs['eval_loss']:.4f}")
                if rec.get("lr") is not None:
                    bits.append(f"lr {rec['lr']:.2e}")
                bits.append(f"{rec['tokens_per_sec']:.1f} tok/s")
                if eta:
                    bits.append(f"ETA {eta / 60:.1f} min")
                bits.append(f"rss {rec['rss_gb']:.2f} GB")
                log(" | ".join(bits))

        def on_train_end(self, args, state, control, **kwargs):
            if self.f is not None:
                self.f.close()
                self.f = None

    return JsonlLogger()


# ---------------------------------------------------------------------------
# Loss-mask verification
# ---------------------------------------------------------------------------


def verify_loss_mask(trainer, tokenizer, dataset, max_print_chars=300):
    """Decode the tokens that actually carry loss for one training example.

    Returns a dict of findings and prints them; the unmasked span must be
    exactly the assistant text plus the end-of-turn token.
    """
    import torch

    example = dataset[0]
    batch = trainer.data_collator([example])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    unmasked_ids = [t for t, l in zip(input_ids, labels) if l != -100]
    masked_ids = [t for t, l in zip(input_ids, labels) if l == -100]
    unmasked_text = tokenizer.decode(unmasked_ids, skip_special_tokens=False)
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=False)
    head_text = tokenizer.decode(input_ids[:12], skip_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    eos_in_span = eos_id in unmasked_ids
    starts_with_bos = bool(input_ids) and input_ids[0] == tokenizer.bos_token_id

    print("=" * 78)
    print("LOSS-MASK VERIFICATION (labels != -100 -> tokens the model is trained on)")
    print("=" * 78)
    print(f"sequence length         : {len(input_ids)} tokens "
          f"({len(unmasked_ids)} unmasked / {len(masked_ids)} masked)")
    print(f"sequence starts with BOS: {starts_with_bos} "
          f"(first ids {input_ids[:4]} -> {head_text!r})")
    print(f"end-of-turn in span     : {eos_in_span} (eos_token={tokenizer.eos_token!r} id={eos_id})")
    print(f"last unmasked ids       : {unmasked_ids[-3:]} -> "
          f"{tokenizer.decode(unmasked_ids[-3:], skip_special_tokens=False)!r}")
    print(f"MASKED (no loss)   [:{max_print_chars}] : {masked_text[:max_print_chars]!r}")
    print(f"UNMASKED (loss)    [:{max_print_chars}] : {unmasked_text[:max_print_chars]!r}")
    print("=" * 78, flush=True)

    return {
        "seq_len": len(input_ids),
        "unmasked_tokens": len(unmasked_ids),
        "masked_tokens": len(masked_ids),
        "starts_with_bos": starts_with_bos,
        "end_of_turn_token_in_unmasked_span": eos_in_span,
        "unmasked_text_head": unmasked_text[:max_print_chars],
        "masked_text_head": masked_text[:max_print_chars],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)

    # Threading env must be set before torch/omp initialise.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    log(f"torch={torch.__version__} threads={torch.get_num_threads()} model={args.model}")

    # ---------------- tokenizer + chat template ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_template = tokenizer.chat_template
    has_markers = bool(model_template) and "{% generation %}" in model_template.replace("{%-", "{%").replace("-%}", "%}")
    chat_template = None  # None => use the tokenizer's own template
    chat_template_source = "model"
    if args.chat_template == "trl-lfm2" or (args.chat_template == "auto" and not has_markers):
        import trl

        tpl_path = os.path.join(os.path.dirname(trl.__file__), "chat_templates", "lfm2_training.jinja")
        with open(tpl_path, "r", encoding="utf-8") as f:
            chat_template = f.read()
        chat_template_source = f"trl:{tpl_path}"
        log(f"using TRL training chat template: {tpl_path}")
    else:
        log("using the model's own chat template (it carries {% generation %} markers)")
    log(f"chat template has generation markers: {has_markers}")

    # ---------------- data ----------------
    train_records = read_jsonl(args.train)
    val_records = read_jsonl(args.val) if args.val else []
    log(f"loaded {len(train_records)} train / {len(val_records)} val records")

    t_tok = time.time()
    train_records, train_stats = measure_and_filter(
        train_records, tokenizer, chat_template, args.max_length, args.long_policy, "train"
    )
    val_stats = None
    if val_records:
        val_records, val_stats = measure_and_filter(
            val_records, tokenizer, chat_template, args.max_length, args.long_policy, "val"
        )
    log(f"tokenized/measured dataset in {time.time() - t_tok:.1f}s")
    log(
        f"train: {train_stats['records_in']} -> {train_stats['records_kept']} kept; "
        f"{train_stats['records_over_max_length']} over --max-length {args.max_length} "
        f"({args.long_policy}); total tokens {train_stats['total_token_stats']['total_tokens']}, "
        f"len p50={train_stats['total_token_stats']['p50']:.0f} p95={train_stats['total_token_stats']['p95']:.0f} "
        f"max={train_stats['total_token_stats']['max']}"
    )
    if val_stats:
        log(
            f"val:   {val_stats['records_in']} -> {val_stats['records_kept']} kept; "
            f"{val_stats['records_over_max_length']} over --max-length"
        )

    if args.max_train_examples is not None:
        train_records = train_records[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_records = val_records[: args.max_val_examples]
    log(f"using {len(train_records)} train / {len(val_records)} val examples")
    if not train_records:
        raise SystemExit("No training examples left after filtering.")

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records])
    eval_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_records]) if val_records else None

    # ---------------- schedule ----------------
    eff_bs = args.per_device_bs * args.grad_accum
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = max(1, round(args.warmup_ratio * total_steps)) if args.warmup_ratio > 0 else 0
    log(
        f"schedule: effective batch {eff_bs} ({args.per_device_bs}x{args.grad_accum}), "
        f"{steps_per_epoch} steps/epoch, {total_steps} total steps, {warmup_steps} warmup steps"
    )

    # ---------------- model ----------------
    # fp32 master weights; bf16 autocast is applied by the Trainer (bf16=True +
    # use_cpu=True), which is what routes matmuls through AMX-BF16.
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.config.use_cache = False
    log(f"base model loaded in {time.time() - t_load:.1f}s, dtype={next(model.parameters()).dtype}, rss={rss_gb():.2f} GB")

    target_modules = args.target_modules
    if target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,  # explicitly do NOT train embed_tokens / lm_head
    )

    gradient_checkpointing = not args.no_gradient_checkpointing
    if gradient_checkpointing:
        # LoRA freezes the base model, so inputs must require grad for
        # checkpointed blocks to build a graph.
        model.enable_input_require_grads()

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        # data / masking
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        chat_template_path=None,  # template handled explicitly below via tokenizer
        dataset_num_proc=None,
        # optimisation
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=warmup_steps,  # transformers 5.x has no warmup_ratio; derived above
        lr_scheduler_type=args.lr_scheduler,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        seed=args.seed,
        data_seed=args.seed,
        # precision / CPU
        bf16=not args.no_bf16,
        fp16=False,
        use_cpu=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # logging / checkpoints
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy if eval_ds is not None else "no",
        eval_steps=args.eval_steps,
        include_num_input_tokens_seen="non_padding",
        report_to=[],
        disable_tqdm=True,
    )
    if chat_template is not None:
        # Applied to the tokenizer itself so trainer preprocessing and the
        # saved tokenizer agree.
        tokenizer.chat_template = chat_template

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    jsonl_logger = make_jsonl_logger_callback(log_path, total_steps)
    callbacks = [jsonl_logger]

    t_trainer = time.time()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    log(f"trainer built in {time.time() - t_trainer:.1f}s")

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    lora_modules = sorted({
        n.split(".lora_A")[0].split(".")[-1]
        for n, _ in trainer.model.named_parameters()
        if "lora_A" in n
    })
    log(f"trainable params {trainable:,} / {total_params:,} ({100 * trainable / total_params:.3f}%)")
    log(f"LoRA target modules resolved: {lora_modules}")
    log(f"mixed precision: {getattr(trainer.accelerator, 'mixed_precision', 'n/a')} | device: {trainer.args.device}")

    # ---------------- loss-mask verification ----------------
    mask_report = None
    if not args.skip_mask_check:
        mask_report = verify_loss_mask(trainer, tokenizer, trainer.train_dataset)
        ref_target = train_records[0]["target"]
        span = mask_report["unmasked_text_head"]
        ok = span.startswith(ref_target[: min(len(ref_target), 240)])
        mask_report["matches_target_prefix"] = bool(ok)
        log(f"mask check: unmasked span starts with the record's target text: {ok}")
        if not mask_report["end_of_turn_token_in_unmasked_span"]:
            log("WARNING: end-of-turn token is NOT inside the unmasked span; the model may never learn to stop.")

    if args.dry_run:
        log("--dry-run set: stopping before training.")
        return 0

    # ---------------- train ----------------
    resume = args.resume_from_checkpoint
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(args.out_dir)
        log(f"resume=auto -> {resume}")

    t_train = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.time() - t_train
    metrics = dict(result.metrics)
    tokens_seen = float(getattr(trainer.state, "num_input_tokens_seen", 0) or 0)
    tokens_this_run = max(tokens_seen - getattr(jsonl_logger, "tokens_at_begin", 0.0), 0.0)
    tok_per_s = tokens_this_run / train_seconds if train_seconds > 0 else 0.0
    log(
        f"training finished in {train_seconds:.1f}s | {tokens_this_run:.0f} tokens this run "
        f"({tokens_seen:.0f} cumulative) | {tok_per_s:.2f} tok/s | peak rss {peak_rss_gb():.2f} GB"
    )

    # ---------------- final eval ----------------
    eval_metrics = {}
    if eval_ds is not None and args.eval_strategy != "no":
        eval_metrics = trainer.evaluate()
        log(f"final eval: {eval_metrics}")

    # ---------------- save ----------------
    adapter_dir = os.path.join(args.out_dir, "adapter")
    trainer.save_model(adapter_dir)  # PEFT model -> adapter weights only
    tokenizer.save_pretrained(adapter_dir)
    log(f"adapter + tokenizer saved to {adapter_dir}")

    base_info = resolve_base_revision(args.model)
    run_config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv),
        "base_model": base_info,
        "base_model_config": {
            "model_type": getattr(model.config, "model_type", None),
            "architectures": getattr(model.config, "architectures", None),
            "vocab_size": getattr(model.config, "vocab_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "commit_hash_from_config": getattr(model.config, "_commit_hash", None),
        },
        "hyperparameters": vars(args),
        "derived": {
            "effective_batch_size": eff_bs,
            "steps_per_epoch": steps_per_epoch,
            "planned_total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "bf16_autocast": not args.no_bf16,
            "mixed_precision": str(getattr(trainer.accelerator, "mixed_precision", "n/a")),
            "chat_template_source": chat_template_source,
            "chat_template_has_generation_markers": has_markers,
            "assistant_only_loss": True,
            "packing": False,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "resolved_modules": lora_modules,
            "trainable_params": trainable,
            "total_params": total_params,
            "trainable_pct": round(100 * trainable / total_params, 4),
            "modules_to_save": None,
        },
        "dataset": {
            "train_path": os.path.abspath(args.train),
            "val_path": os.path.abspath(args.val) if args.val else None,
            "train_used": len(train_ds),
            "val_used": len(eval_ds) if eval_ds is not None else 0,
            "train_stats": train_stats,
            "val_stats": val_stats,
        },
        "loss_mask_verification": mask_report,
        "results": {
            "train_metrics": metrics,
            "eval_metrics": eval_metrics,
            "train_seconds": round(train_seconds, 2),
            "tokens_seen_non_padding": int(tokens_seen),
            "tokens_seen_this_run": int(tokens_this_run),
            "tokens_per_second": round(tok_per_s, 2),
            "peak_rss_gb": round(peak_rss_gb(), 3),
            "wall_seconds_total": round(time.time() - t_start, 2),
        },
        "versions": {},
    }
    try:
        import datasets as _ds
        import peft as _peft
        import transformers as _tf
        import trl as _trl

        run_config["versions"] = {
            "torch": torch.__version__,
            "transformers": _tf.__version__,
            "peft": _peft.__version__,
            "trl": _trl.__version__,
            "datasets": _ds.__version__,
        }
    except Exception:
        pass

    cfg_path = os.path.join(args.out_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)
    log(f"run config written to {cfg_path}")
    log(f"DONE in {time.time() - t_start:.1f}s | peak rss {peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
