#!/usr/bin/env python3
"""CPU-only LoRA fine-tuning throughput benchmark.

Measures, per candidate base model, on THIS machine:
  - model load time (bf16) and peak process RSS
  - LoRA (r=16 / alpha=32 / dropout=0.05, all linear projections) train
    throughput: 1 warm-up step + 3 timed AdamW steps on a synthetic
    1x512-token batch (forward + backward + optimizer step)
  - plain greedy generation throughput on the base model (200-token
    prompt, 128 new tokens) -- this is the eval-speed number
  - (LFM2 only) full fine-tuning throughput: fp32 master weights, bf16
    autocast compute, every parameter trainable, no LoRA

Design notes
------------
* Every (model, mode) benchmark runs in its OWN subprocess ("worker"), so
  a crash or an OS OOM-kill for one model cannot lose results already
  collected for earlier models, and cannot corrupt a later model's run
  (fresh process = fresh, unfragmented address space, fresh thread pool).
* Workers append markdown to the results file incrementally, after each
  phase (load / generation / train), so even a hard kill mid-benchmark
  leaves whatever was already measured on disk.
* A background thread polls resource.getrusage().ru_maxrss during the
  run and self-terminates the process if RSS crosses a configurable
  guard threshold, so a near-OOM run leaves a clean recorded result
  instead of an untraceable SIGKILL from the kernel OOM-killer.
* Model loading tries progressively more manual strategies so that
  multimodal checkpoints (Qwen3.5, Gemma4) load as text-only causal LMs
  wherever the installed transformers version supports it (skipping the
  vision/audio towers), and only falls back to the full
  conditional-generation model as a last resort.

Usage
-----
    # Full sweep across all candidate models (spawns one subprocess per
    # (model, mode) run; this is what actually produced bench_results.md):
    python3 bench_throughput.py

    # Single benchmark run (what the sweep spawns internally):
    python3 bench_throughput.py --worker --repo LiquidAI/LFM2.5-350M \
        --tag lfm2 --mode lora
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()

DEFAULT_RESULTS_PATH = (
    "/tmp/claude-0/-home-user-copywritingModel/"
    "f813ae63-f377-5d66-8e03-fd0bcbb1e21f/scratchpad/bench_results.md"
)

SEQ_LEN = 512
GEN_PROMPT_LEN = 200
GEN_NEW_TOKENS = 128
WARMUP_STEPS = 1
TIMED_STEPS = 3
NUM_THREADS = 4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

KB_PER_GB = 1024.0 * 1024.0

# This sandbox injects the HF token as HUGGINGFACE_TOKEN; huggingface_hub
# only auto-picks-up HF_TOKEN / HUGGING_FACE_HUB_TOKEN, so normalize it
# before any huggingface_hub/transformers code runs.
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)

MODEL_SPECS = [
    {
        "tag": "lfm2",
        "repo": "LiquidAI/LFM2.5-350M",
        "approx_gb": 0.7,
        "rss_guard_gb": None,
        "also_full_finetune": True,
    },
    {
        "tag": "qwen35",
        "repo": "Qwen/Qwen3.5-0.8B",
        "approx_gb": 1.75,
        "rss_guard_gb": None,
        "also_full_finetune": False,
    },
    {
        "tag": "gemma4",
        "repo": "google/gemma-4-E2B-it",
        "approx_gb": 10.25,
        "rss_guard_gb": 13.0,
        "min_free_disk_gb": 12.0,
        "min_free_ram_gb": 10.0,
        "also_full_finetune": False,
    },
]


# --------------------------------------------------------------------------
# Small utilities shared by orchestrator + worker
# --------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rss_gb() -> float:
    """Peak RSS of this process so far, in GiB (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB_PER_GB


def disk_free_gb(path: str) -> float:
    while path and not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path or "/").free / (1024.0 ** 3)


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 ** 2)
    except OSError:
        pass
    return None


def append_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def status_path(status_dir: str, tag: str, mode: str) -> str:
    return os.path.join(status_dir, f"{tag}_{mode}.json")


def hf_cache_root() -> str:
    return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")


def delete_from_hf_cache(repo_id: str) -> str:
    """Best-effort removal of `repo_id` from the local HF cache. Returns a note."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        hashes = []
        for repo in info.repos:
            if repo.repo_id == repo_id:
                hashes.extend(rev.commit_hash for rev in repo.revisions)
        if not hashes:
            return f"nothing to delete for {repo_id} (not found in cache scan)"
        strategy = info.delete_revisions(*hashes)
        freed_gb = strategy.expected_freed_size / (1024.0 ** 3)
        strategy.execute()
        return f"deleted {repo_id} from HF cache, freed ~{freed_gb:.2f} GB"
    except Exception as e:  # noqa: BLE001 - best-effort cleanup
        try:
            safe = repo_id.replace("/", "--")
            folder = os.path.join(hf_cache_root(), "hub", f"models--{safe}")
            if os.path.isdir(folder):
                shutil.rmtree(folder)
                return f"deleted {repo_id} via raw folder removal ({folder})"
            return f"cache cleanup failed for {repo_id}: {e!r}; no matching folder either"
        except Exception as e2:  # noqa: BLE001
            return f"cache cleanup failed for {repo_id}: {e!r} / fallback also failed: {e2!r}"


# --------------------------------------------------------------------------
# RSS guard: background watchdog that self-terminates the process if peak
# RSS crosses a threshold, so a near-OOM run leaves a clean recorded result
# instead of a bare SIGKILL from the kernel.
# --------------------------------------------------------------------------

class RssGuard:
    def __init__(self, threshold_gb: float, on_trip, poll_s: float = 0.5):
        self.threshold_gb = threshold_gb
        self.on_trip = on_trip
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.current_phase = "startup"

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            current = rss_gb()
            if current >= self.threshold_gb:
                try:
                    self.on_trip(current, self.current_phase)
                finally:
                    os._exit(1)  # hard exit now; no further allocation risk
            self._stop.wait(self.poll_s)


# --------------------------------------------------------------------------
# Model loading: prefer a text-only causal-LM view of multimodal
# checkpoints (skips vision/audio towers), fall back to the full
# conditional-generation / omni model as a last resort.
# --------------------------------------------------------------------------

class LoadFailure(RuntimeError):
    pass


def _text_config_of(config):
    """Return the nested text sub-config if this is a multimodal composite
    config, else None (mirrors `config.get_text_config() is config`)."""
    if hasattr(config, "get_text_config"):
        try:
            tc = config.get_text_config()
            if tc is not None and tc is not config:
                return tc
        except Exception:
            pass
    return None


def _from_pretrained(auto_cls, repo, dtype, token, low_cpu_mem_usage, **extra):
    kwargs = dict(low_cpu_mem_usage=low_cpu_mem_usage, token=token, **extra)
    try:
        return auto_cls.from_pretrained(repo, torch_dtype=dtype, **kwargs)
    except TypeError:
        # transformers versions differ on the `torch_dtype` vs `dtype` kwarg name.
        return auto_cls.from_pretrained(repo, dtype=dtype, **kwargs)


def load_causal_lm(repo: str, dtype, token, low_cpu_mem_usage=True):
    """Return (model, strategy_name, attempt_errors)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    attempt_errors = []
    config = AutoConfig.from_pretrained(repo, token=token)
    text_config = _text_config_of(config)

    # Preferred order: if this is a multimodal composite checkpoint, try the
    # *text-only* sub-config FIRST -- AutoModelForCausalLM resolves that to
    # the text-only class (e.g. Gemma4ForCausalLM / Qwen3_5ForCausalLM)
    # registered under the text sub-config's model_type, so the vision/audio
    # tower submodules are never constructed or allocated. Only fall back to
    # the checkpoint's own top-level config (which for a multimodal
    # checkpoint resolves to the full *ForConditionalGeneration class) if
    # that fails.
    ordered_configs = []
    if text_config is not None:
        mt = getattr(text_config, "model_type", "?")
        ordered_configs.append((f"AutoModelForCausalLM(text_config, model_type={mt})", text_config))
    mt0 = getattr(config, "model_type", "?")
    ordered_configs.append((f"AutoModelForCausalLM(default config, model_type={mt0})", config))

    for label, cfg in ordered_configs:
        try:
            model = _from_pretrained(AutoModelForCausalLM, repo, dtype, token, low_cpu_mem_usage, config=cfg)
            return model, label, attempt_errors
        except Exception as e:
            attempt_errors.append((label, repr(e)))

    # Last resort: load the full multimodal/omni model. Vision/audio tower
    # weights DO end up resident in RAM here, but forward() on these models
    # runs the text-only path whenever pixel_values / input_features aren't
    # supplied, so training/generation are still computed correctly.
    for cls_name in ("AutoModelForImageTextToText", "AutoModel"):
        try:
            cls = getattr(__import__("transformers", fromlist=[cls_name]), cls_name)
        except AttributeError:
            continue
        try:
            model = _from_pretrained(cls, repo, dtype, token, low_cpu_mem_usage)
            return model, f"{cls_name}(full multimodal model, text-only forward)", attempt_errors
        except Exception as e:
            attempt_errors.append((cls_name, repr(e)))

    raise LoadFailure("all loading strategies failed:\n" + json.dumps(attempt_errors, indent=2))


def get_vocab_size(model) -> int:
    cfg = model.config
    tc = _text_config_of(cfg) or cfg
    vs = getattr(tc, "vocab_size", None)
    if vs is None:
        vs = getattr(cfg, "vocab_size")
    return int(vs)


def is_probably_text_path(dotted_name: str) -> bool:
    """Filters out vision/audio-tower submodules by name, in case the
    fallback "full multimodal model" loading strategy was used and its
    parameter tree includes non-text towers."""
    lowered = dotted_name.lower()
    blocked = ("vision", "visual", "image_encoder", "audio", "speech", "multi_modal_projector", "mm_projector")
    return not any(b in lowered for b in blocked)


def discover_target_modules(model, name_filter=None):
    import torch.nn as nn

    leaf_counts: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name_filter is not None and not name_filter(name):
                continue
            leaf = name.rsplit(".", 1)[-1]
            leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

    exclude = {"lm_head"}
    try:
        out_emb = model.get_output_embeddings()
    except Exception:
        out_emb = None
    if out_emb is not None:
        for name, module in model.named_modules():
            if module is out_emb:
                exclude.add(name.rsplit(".", 1)[-1])

    targets = sorted(n for n in leaf_counts if n not in exclude)
    return targets, leaf_counts, sorted(exclude)


# --------------------------------------------------------------------------
# Benchmark phases
# --------------------------------------------------------------------------

def build_batch(vocab_size: int, seq_len: int):
    import torch

    return torch.randint(low=0, high=vocab_size, size=(1, seq_len), dtype=torch.long)


def run_generation_bench(model, vocab_size: int, prompt_len: int, new_tokens: int):
    import torch

    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    prompt = build_batch(vocab_size, prompt_len)
    pad_id = getattr(model.config, "pad_token_id", None) or getattr(model.config, "eos_token_id", None) or 0
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(
            prompt,
            min_new_tokens=new_tokens,
            max_new_tokens=new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=pad_id,
        )
        t1 = time.perf_counter()
    actual_new = out.shape[1] - prompt.shape[1]
    elapsed = t1 - t0
    return {
        "prompt_len": prompt_len,
        "requested_new_tokens": new_tokens,
        "actual_new_tokens": int(actual_new),
        "elapsed_s": elapsed,
        "tokens_per_sec": actual_new / elapsed if elapsed > 0 else float("nan"),
    }


def enable_gradient_checkpointing(model):
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def compute_loss(model, input_ids, labels):
    import torch.nn.functional as F

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = getattr(out, "loss", None)
    if loss is not None:
        return loss
    logits = out.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def run_train_bench(model, optimizer, vocab_size, seq_len, warmup_steps, timed_steps,
                     autocast_ctx=None, phase_cb=None):
    model.train()
    input_ids = build_batch(vocab_size, seq_len)
    labels = input_ids.clone()

    def step():
        optimizer.zero_grad(set_to_none=True)
        ctx = autocast_ctx() if autocast_ctx is not None else contextlib.nullcontext()
        with ctx:
            loss = compute_loss(model, input_ids, labels)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for i in range(warmup_steps):
        if phase_cb:
            phase_cb(f"train_warmup_{i}")
        step()

    step_times = []
    last_loss = None
    for i in range(timed_steps):
        if phase_cb:
            phase_cb(f"train_timed_{i}")
        t0 = time.perf_counter()
        last_loss = step()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    total_time = sum(step_times)
    total_tokens = timed_steps * seq_len
    return {
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "seq_len": seq_len,
        "step_times_s": step_times,
        "total_time_s": total_time,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else float("nan"),
        "last_loss": last_loss,
    }


# --------------------------------------------------------------------------
# Worker: benchmarks exactly one (repo, mode) combination in this process.
# --------------------------------------------------------------------------

def worker_main(args) -> int:
    import torch

    torch.set_num_threads(args.threads)

    results_path = args.results_path
    status_dir = args.status_dir
    tag, mode, repo = args.tag, args.mode, args.repo
    spath = status_path(status_dir, tag, mode)

    def log(msg):
        print(f"[worker:{tag}:{mode}] {msg}", flush=True)

    append_text(results_path, f"\n## {repo} -- mode={mode} (tag={tag})\nstarted: {now_iso()}\n")

    status = {"repo": repo, "tag": tag, "mode": mode, "started": now_iso(), "status": "RUNNING"}
    write_json(spath, status)

    guard = None

    def on_trip(current_gb, phase):
        append_text(
            results_path,
            f"\n**ABORTED (RSS guard tripped)** -- phase=`{phase}`, "
            f"rss={current_gb:.2f} GB >= threshold={args.rss_guard_gb:.2f} GB\n",
        )
        status.update(status="OOM_GUARD", phase_at_abort=phase, rss_gb_at_abort=current_gb)
        write_json(spath, status)
        log(f"RSS GUARD TRIPPED at phase={phase}, rss={current_gb:.2f} GB")

    if args.rss_guard_gb:
        guard = RssGuard(args.rss_guard_gb, on_trip).start()

    def set_phase(p):
        log(f"phase={p}")
        if guard:
            guard.current_phase = p

    try:
        dtype = torch.float32 if mode == "full" else torch.bfloat16

        set_phase("load")
        t0 = time.perf_counter()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)
        model, strategy, load_errors = load_causal_lm(repo, dtype, HF_TOKEN)
        t1 = time.perf_counter()
        load_time_s = t1 - t0
        vocab_size = get_vocab_size(model)
        n_params = sum(p.numel() for p in model.parameters())

        load_block = (
            f"- load_time_s: {load_time_s:.2f}\n"
            f"- load_strategy: {strategy}\n"
            f"- dtype: {dtype}\n"
            f"- total_params: {n_params:,}\n"
            f"- vocab_size: {vocab_size}\n"
            f"- rss_gb_after_load: {rss_gb():.2f}\n"
        )
        if load_errors:
            load_block += f"- load_fallback_errors: {json.dumps(load_errors)}\n"
        append_text(results_path, load_block)
        status.update(load_time_s=load_time_s, load_strategy=strategy, total_params=n_params)
        write_json(spath, status)
        log(f"loaded in {load_time_s:.2f}s via {strategy}, {n_params:,} params, rss={rss_gb():.2f} GB")

        gen_result = None
        if mode == "lora":
            set_phase("generation")
            gen_result = run_generation_bench(model, vocab_size, args.gen_prompt_len, args.gen_new_tokens)
            append_text(
                results_path,
                f"- gen_prompt_len: {gen_result['prompt_len']}\n"
                f"- gen_new_tokens: {gen_result['actual_new_tokens']}\n"
                f"- gen_time_s: {gen_result['elapsed_s']:.2f}\n"
                f"- gen_tokens_per_sec: {gen_result['tokens_per_sec']:.2f}\n"
                f"- rss_gb_after_gen: {rss_gb():.2f}\n",
            )
            status.update(gen_tokens_per_sec=gen_result["tokens_per_sec"])
            write_json(spath, status)
            log(f"generation: {gen_result['tokens_per_sec']:.2f} tok/s, rss={rss_gb():.2f} GB")

        set_phase("train_setup")
        enable_gradient_checkpointing(model)

        if mode == "lora":
            targets, leaf_counts, excluded = discover_target_modules(model, name_filter=is_probably_text_path)
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=targets,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            trainable = [p for p in model.parameters() if p.requires_grad]
            n_trainable = sum(p.numel() for p in trainable)
            append_text(
                results_path,
                f"- discovered_linear_leaf_names: {json.dumps(leaf_counts)}\n"
                f"- excluded_from_lora: {excluded}\n"
                f"- lora_target_modules: {targets}\n"
                f"- lora_r_alpha_dropout: {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}\n"
                f"- trainable_params: {n_trainable:,} ({100.0 * n_trainable / n_params:.3f}% of base)\n",
            )
            log(f"LoRA target_modules={targets}, trainable={n_trainable:,}")
        else:
            trainable = list(model.parameters())
            for p in trainable:
                p.requires_grad_(True)
            n_trainable = sum(p.numel() for p in trainable)
            append_text(results_path, f"- full_finetune_trainable_params: {n_trainable:,}\n")
            log(f"full fine-tune, trainable={n_trainable:,}")

        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        autocast_ctx = None
        if mode == "full":
            autocast_ctx = lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16)  # noqa: E731

        set_phase("train")
        train_result = run_train_bench(
            model, optimizer, vocab_size, args.seq_len, args.warmup_steps, args.timed_steps,
            autocast_ctx=autocast_ctx, phase_cb=set_phase,
        )
        append_text(
            results_path,
            f"- train_warmup_steps: {train_result['warmup_steps']}\n"
            f"- train_timed_steps: {train_result['timed_steps']}\n"
            f"- train_seq_len: {train_result['seq_len']}\n"
            f"- train_step_times_s: {[round(t, 3) for t in train_result['step_times_s']]}\n"
            f"- train_tokens_per_sec: {train_result['tokens_per_sec']:.2f}\n"
            f"- train_last_loss: {train_result['last_loss']:.4f}\n"
            f"- rss_gb_after_train: {rss_gb():.2f}\n",
        )
        log(f"train: {train_result['tokens_per_sec']:.2f} tok/s, rss={rss_gb():.2f} GB")

        peak_rss = rss_gb()
        status.update(status="OK", train_tokens_per_sec=train_result["tokens_per_sec"], peak_rss_gb=peak_rss)
        write_json(spath, status)

        append_text(
            results_path,
            f"\n**SUMMARY** {repo} [{mode}]: status=OK, load_s={load_time_s:.2f}, "
            f"train_tok_s={train_result['tokens_per_sec']:.2f}, "
            + (f"gen_tok_s={gen_result['tokens_per_sec']:.2f}, " if gen_result else "")
            + f"peak_rss_gb={peak_rss:.2f}\n",
        )
        log(f"DONE status=OK peak_rss={peak_rss:.2f} GB")

        if guard:
            guard.stop()
        return 0

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        peak_rss = rss_gb()
        append_text(
            results_path,
            f"\n**FAILURE** {repo} [{mode}]: status=FAILED, peak_rss_gb={peak_rss:.2f}\n```\n{tb}\n```\n",
        )
        status.update(status="FAILED", error=tb, peak_rss_gb=peak_rss)
        write_json(spath, status)
        log(f"DONE status=FAILED peak_rss={peak_rss:.2f} GB")
        if guard:
            guard.stop()
        return 1


# --------------------------------------------------------------------------
# Orchestrator: runs the full sweep, one subprocess per (model, mode).
# --------------------------------------------------------------------------

def spawn_worker(repo, tag, mode, results_path, status_dir, rss_guard_gb, threads, timeout_s):
    cmd = [
        sys.executable, str(THIS_FILE), "--worker",
        "--repo", repo, "--tag", tag, "--mode", mode,
        "--results-path", results_path, "--status-dir", status_dir,
        "--threads", str(threads),
    ]
    if rss_guard_gb:
        cmd += ["--rss-guard-gb", str(rss_guard_gb)]
    print(f"[orchestrator] launching: {' '.join(cmd)} (timeout={timeout_s}s)", flush=True)
    try:
        proc = subprocess.run(cmd, timeout=timeout_s)
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        return None, True


def handle_worker_outcome(repo, mode, results_path, status_dir, tag, returncode, timed_out):
    spath = status_path(status_dir, tag, mode)
    status = read_json(spath)
    if timed_out:
        append_text(
            results_path,
            f"\n**ORCHESTRATOR NOTE** {repo} [{mode}]: worker subprocess timed out and was killed.\n",
        )
        return status or {"status": "TIMEOUT"}
    if status is not None:
        return status
    append_text(
        results_path,
        f"\n**ORCHESTRATOR NOTE** {repo} [{mode}]: worker exited with code {returncode} "
        f"and left no status file (likely killed by the OS, e.g. OOM-killer).\n",
    )
    return {"status": "KILLED", "returncode": returncode}


def orchestrate(args):
    results_path = args.results_path
    status_dir = args.status_dir
    Path(status_dir).mkdir(parents=True, exist_ok=True)

    append_text(
        results_path,
        f"\n# Bench run started {now_iso()}\n"
        f"threads={args.threads}, seq_len={args.seq_len}, "
        f"gen_prompt_len={args.gen_prompt_len}, gen_new_tokens={args.gen_new_tokens}\n",
    )

    for spec in MODEL_SPECS:
        repo, tag = spec["repo"], spec["tag"]

        if spec.get("min_free_disk_gb") or spec.get("min_free_ram_gb"):
            free_disk = disk_free_gb(hf_cache_root())
            free_mem = mem_available_gb()
            need_disk = spec.get("min_free_disk_gb", 0)
            need_mem = spec.get("min_free_ram_gb", 0)
            disk_ok = free_disk >= need_disk
            mem_ok = free_mem is None or free_mem >= need_mem
            if not (disk_ok and mem_ok):
                append_text(
                    results_path,
                    f"\n## {repo} -- SKIPPED before download\n"
                    f"free_disk_gb={free_disk:.1f} (need >={need_disk}), "
                    f"free_mem_gb={free_mem}, (need >={need_mem}). Not downloading.\n",
                )
                print(f"[orchestrator] skipping {repo}: disk_ok={disk_ok} mem_ok={mem_ok}", flush=True)
                continue
            print(
                f"[orchestrator] precheck OK for {repo}: {free_disk:.1f} GB free disk, "
                f"{free_mem} GB mem available",
                flush=True,
            )

        timeout_s = max(900, int(spec["approx_gb"] * 240) + 900)
        returncode, timed_out = spawn_worker(
            repo, tag, "lora", results_path, status_dir, spec.get("rss_guard_gb"), args.threads, timeout_s,
        )
        status = handle_worker_outcome(repo, "lora", results_path, status_dir, tag, returncode, timed_out)
        print(f"[orchestrator] {repo} [lora] -> {status.get('status')}", flush=True)

        if spec.get("also_full_finetune"):
            returncode2, timed_out2 = spawn_worker(
                repo, tag, "full", results_path, status_dir, spec.get("rss_guard_gb"), args.threads, timeout_s,
            )
            status2 = handle_worker_outcome(repo, "full", results_path, status_dir, tag, returncode2, timed_out2)
            print(f"[orchestrator] {repo} [full] -> {status2.get('status')}", flush=True)

        if spec.get("min_free_disk_gb"):  # i.e. the one big/risky model we prechecked
            viable = status.get("status") == "OK"
            note = delete_from_hf_cache(repo) if not viable else f"kept {repo} in HF cache (viable)"
            append_text(results_path, f"\n**CACHE CLEANUP** {repo}: viable={viable}. {note}\n")
            print(f"[orchestrator] {repo}: viable={viable}. {note}", flush=True)

    append_text(results_path, f"\n# Bench run finished {now_iso()}\n")
    print("[orchestrator] done.", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help="Run a single (repo, mode) benchmark in this process.")
    p.add_argument("--repo", type=str, help="HF repo id (worker mode).")
    p.add_argument("--tag", type=str, help="Short label for this model (worker mode).")
    p.add_argument("--mode", type=str, choices=["lora", "full"], default="lora")
    p.add_argument("--results-path", type=str, default=DEFAULT_RESULTS_PATH)
    p.add_argument("--status-dir", type=str, default=None)
    p.add_argument("--rss-guard-gb", type=float, default=None)
    p.add_argument("--threads", type=int, default=NUM_THREADS)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--gen-prompt-len", type=int, default=GEN_PROMPT_LEN)
    p.add_argument("--gen-new-tokens", type=int, default=GEN_NEW_TOKENS)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--timed-steps", type=int, default=TIMED_STEPS)
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.status_dir is None:
        args.status_dir = str(Path(args.results_path).parent / "bench_status")

    if args.worker:
        if not args.repo or not args.tag:
            print("--worker requires --repo and --tag", file=sys.stderr)
            sys.exit(2)
        sys.exit(worker_main(args))
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
