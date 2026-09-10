"""Shared helpers for the landing-page copywriting evaluation harness.

Covers: dataset I/O, chat-template formatting that works across chat
templates with different quirks (system-role support, `enable_thinking`,
<think> blocks), model/tokenizer/adapter loading, and tag-line parsing for
the "one element per line" target format.

Kept dependency-light (stdlib + torch/transformers/peft only) so every
script in scripts/ can import it identically.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Imported from prompt.py (owned by another agent) so every script always
# uses the exact same constant. Fall back to a local copy only if the import
# fails outright (e.g. script run from outside scripts/ with a broken path),
# so the eval harness never hard-crashes on an import wrinkle.
try:
    from prompt import SYSTEM_PROMPT
except ImportError:  # pragma: no cover - defensive fallback only
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from prompt import SYSTEM_PROMPT
    except ImportError:
        SYSTEM_PROMPT = (
            "You are an expert landing page copywriter. Given a product brief, "
            "write the complete landing page copy. Output one element per line "
            "using only these tags: <h1>, <h2>, <h3>, <h4>, <p>, <button>. Use "
            "exactly one <h1> for the hero headline, <h2> for section headlines, "
            "<h3> and <h4> for feature or benefit titles, <p> for supporting "
            "copy, and <button> for call-to-action labels. Write in page order "
            "and output nothing else."
        )

TAG_NAMES = ("h1", "h2", "h3", "h4", "p", "button")

# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: str, n: Optional[int] = None) -> list[dict]:
    """Read a JSONL file, optionally truncated to the first n records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if n is not None and len(records) >= n:
                break
    return records


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def existing_ids(path: str) -> set:
    """IDs already present in an output JSONL file, for resumable runs."""
    ids = set()
    if not os.path.exists(path):
        return ids
    for rec in iter_jsonl(path):
        if "id" in rec:
            ids.add(rec["id"])
    return ids


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------


def build_messages(brief: str, target: Optional[str] = None, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Build a fresh messages list in the dataset's schema."""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": brief},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


def get_messages(record: dict, include_assistant: bool = True) -> list[dict]:
    """Get a record's chat messages, preferring the dataset's own "messages"
    field (so we stay faithful to whatever build_dataset.py produced) and
    falling back to constructing them from brief/target with SYSTEM_PROMPT.
    """
    msgs = record.get("messages")
    if not msgs:
        msgs = build_messages(record["brief"], record.get("target"))
    msgs = [dict(m) for m in msgs]
    if not include_assistant:
        msgs = [m for m in msgs if m.get("role") != "assistant"]
    return msgs


def fold_system_into_user(messages: list[dict]) -> list[dict]:
    """Fold system message(s) into the first non-system turn's content.

    Used when a chat template raises on a system role (e.g. classic Gemma
    templates). Order of remaining messages is preserved.
    """
    sys_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if not sys_parts or not rest:
        return rest or [dict(m) for m in messages]
    rest[0] = dict(rest[0])
    rest[0]["content"] = "\n\n".join(sys_parts) + "\n\n" + rest[0]["content"]
    return rest


def apply_chat_template_safe(
    tokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    enable_thinking: bool = False,
):
    """tokenizer.apply_chat_template that works across model families.

    - Passes enable_thinking=False so templates that support it (Qwen-style)
      don't inject a <think> scaffold; falls back to omitting the kwarg via
      try/except TypeError for templates whose signature rejects it.
    - If the template raises on a system-role message (classic Gemma-style
      templates), folds the system prompt into the first user turn and
      retries once.
    - When tokenize=True, always returns a flat list[int] (this transformers
      version defaults tokenize=True to return_dict=True, which would
      otherwise hand back a BatchEncoding instead of ids).
    """

    def _try(msgs):
        try:
            return tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                return_dict=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                return_dict=False,
            )

    try:
        return _try(messages)
    except TypeError:
        raise
    except Exception:
        folded = fold_system_into_user(messages)
        try:
            return _try(folded)
        except Exception as e2:
            raise RuntimeError(
                "apply_chat_template failed both with a system-role message "
                f"and after folding it into the user turn: {e2}"
            ) from e2


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from a generation.

    Also strips a dangling, unclosed <think> (generation cut off mid-block
    by max_new_tokens) through end-of-string.
    """
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    return text.strip()


def _prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def assistant_token_span(tokenizer, record: dict, enable_thinking: bool = False) -> tuple[list[int], int]:
    """Tokenize a record's full conversation and locate where the assistant
    span starts, per the spec's method: tokenize the prompt-only chat
    template (add_generation_prompt=True) and take the suffix of the full
    conversation's tokenization as the assistant span.

    Returns (full_ids, prompt_len) where full_ids[prompt_len:] are the
    assistant (target) tokens, including the trailing EOS the template adds.
    """
    full_messages = get_messages(record, include_assistant=True)
    prompt_messages = get_messages(record, include_assistant=False)

    full_ids = apply_chat_template_safe(
        tokenizer, full_messages, add_generation_prompt=False, tokenize=True, enable_thinking=enable_thinking
    )
    prompt_ids = apply_chat_template_safe(
        tokenizer, prompt_messages, add_generation_prompt=True, tokenize=True, enable_thinking=enable_thinking
    )
    # prompt_ids should be an exact prefix of full_ids; fall back to the
    # longest common prefix if a template tokenizes the boundary slightly
    # differently depending on what follows (rare, but cheap to guard).
    prompt_len = _prefix_len(prompt_ids, full_ids)
    if prompt_len < len(prompt_ids):
        prompt_len = len(prompt_ids)
        prompt_len = min(prompt_len, len(full_ids))
    return full_ids, prompt_len


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_name: str, adapter: Optional[str] = None, dtype=None, threads: Optional[int] = None):
    """Load an AutoTokenizer + AutoModelForCausalLM pair (CPU, bf16 by
    default), optionally wrapping the base model with a PEFT adapter. Fully
    model-agnostic: works for any of the three candidate base models.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if threads:
        torch.set_num_threads(threads)
    if dtype is None:
        dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to("cpu")
    model.eval()
    return model, tokenizer


def eos_token_ids(tokenizer, model) -> set:
    """Collect every token id that should count as an end-of-sequence
    marker, from both the tokenizer and the model's generation_config
    (some chat models add extra stop tokens like <|im_end|>).
    """
    ids = set()
    for src in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if src is None:
            continue
        if isinstance(src, (list, tuple, set)):
            ids.update(int(x) for x in src)
        else:
            ids.add(int(src))
    return ids


# ---------------------------------------------------------------------------
# Tag-line parsing (the "one element per line" target/prediction format)
# ---------------------------------------------------------------------------

LINE_RE = re.compile(r"^<(h1|h2|h3|h4|p|button)>(.+)</\1>$")


@dataclass
class ParsedLine:
    raw: str
    tag: Optional[str]
    content: Optional[str]
    valid: bool


def parse_lines(text: str) -> list[ParsedLine]:
    """Parse every non-blank line of text against the tag-line format.

    Blank lines are dropped (formatting noise, not content); every other
    line is kept, matched or not, so callers can compute both "does every
    line match" checks and "what tags/content are actually present" stats
    from the same parse.
    """
    out = []
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped == "":
            continue
        m = LINE_RE.match(stripped)
        if m:
            out.append(ParsedLine(raw=stripped, tag=m.group(1), content=m.group(2), valid=True))
        else:
            out.append(ParsedLine(raw=stripped, tag=None, content=None, valid=False))
    return out


def valid_tag_lines(text: str) -> list[ParsedLine]:
    """Only the lines that matched the tag-line format."""
    return [pl for pl in parse_lines(text) if pl.valid]


def content_text(text: str) -> str:
    """The visible copy only: tag contents of valid lines joined by spaces,
    with all markup stripped. Tag names like <h1>/<h2> contain digits and
    letters that would otherwise contaminate word/number-based metrics
    (e.g. distinct-n, hallucinated-number counting) if run on raw text.
    """
    return " ".join(pl.content for pl in valid_tag_lines(text))


# ---------------------------------------------------------------------------
# Generic text utilities
# ---------------------------------------------------------------------------

WORD_RE = re.compile(r"[A-Za-z']+")
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?%?")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about
    against between into through during before after above below to from
    up down in out on off over under again further once here there all
    any both each few more most other some such no nor not only own same
    so than too very s t can will just don should now is are was were be
    been being have has had having do does did doing would could might
    must shall this that these those i you he she it we they me him her
    them my your his its our their what which who whom as it's your our
    us get gets getting new best top use uses using product brief
    """.split()
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 2]


def product_name(brief: str) -> Optional[str]:
    """Extract the product name from a brief's "Product:" line."""
    for line in brief.splitlines():
        line = line.strip()
        if line.lower().startswith("product:"):
            name = line.split(":", 1)[1].strip()
            return name or None
    return None


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def js_divergence(p: list[float], q: list[float], base: float = 2.0) -> float:
    """Jensen-Shannon divergence between two discrete distributions given
    as raw (unnormalized) vectors of equal length. No scipy dependency.
    """
    import numpy as np

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])) / np.log(base))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
