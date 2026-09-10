#!/usr/bin/env python3
"""curate_dataset.py

Curates the landing-page copywriting dataset (train/val/test JSONL) into a
quality-filtered, re-split copy.

Pipeline (applied identically to train.jsonl, val.jsonl, test.jsonl):
  1. Parse every line of `target` as a `<tag>text</tag>` element (h1/h2/h3/
     h4/p/button). Any record with a line that fails to parse is dropped.
  1.5. Per-line filters, applied before the record-level rules:
       (a) drop a <p> line only if it starts with a lowercase letter, or
           starts with a character that isn't a letter/digit/quote/opening
           parenthesis.
       (b) drop a <button>/heading line only if it has >= 4 words AND
           (contains a non-ASCII letter OR contains >= 2 words from a small
           Romance/Germanic/Dutch function-word list) AND langdetect is
           confident (>0.9) it's not English (langdetect calls are wrapped
           in try/except; lines that don't meet the word/vocabulary gate are
           never run through langdetect at all).
       (c) if a record loses more than 5 lines, or more than 25% of its
           lines, to (a)+(b) combined, the whole record is dropped instead
           of being trimmed.
  2. Normalize the hero: the single <h1> must appear within the first three
     elements, else the record is dropped; if it is not already first, it is
     moved to position 0 (all other elements keep their relative order).
  3. Keep only records that satisfy a battery of landing-page-copy quality
     rules (see `check_quality` / `RULE_DESCRIPTIONS` below). The first rule
     a record fails is recorded as its drop reason.
  4. Rebuild `target` and `messages[2]["content"]` from the normalized
     element list (ids and all other fields are left untouched).
  5. Re-split: curated val/test records are kept as-is, then topped up from
     curated train records (random, seed 42) -- picking train records whose
     URL domain is not already present in val/test -- until val reaches
     --target-val and test reaches --target-test. Everything else is train.
  6. Write train.jsonl / val.jsonl / test.jsonl plus stats.md to the output
     directory.

Usage:
    python3 curate_dataset.py [--input-dir DIR] [--output-dir DIR]
                               [--seed 42] [--target-val 100] [--target-test 120]
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

try:
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - guarded import
    detect_langs = None
    LangDetectException = Exception

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = (
    "/tmp/claude-0/-home-user-copywritingModel/"
    "f813ae63-f377-5d66-8e03-fd0bcbb1e21f/scratchpad/data/dataset"
)

TAGS = ("h1", "h2", "h3", "h4", "p", "button")
TAG_RE = re.compile(r"^<(h1|h2|h3|h4|p|button)>(.*)</\1>$")

BANNED_PHRASES = ("cookie", "javascript", "subscribe to our newsletter")

MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
}

_ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")
_DATE_SEP_RE = re.compile(r"^\d{1,4}[/\-:]\d{1,4}([/\-:]\d{1,4})?$")
_NUMERIC_RE = re.compile(
    r"^[$€£¥]?\d[\d,]*(\.\d+)?(%|\+|x)?"
    r"(/(mo|month|yr|year|wk|week|day|hr|hour))?[kmb]?\+?$"
)

# Per-line filter constants (rules a/b, applied before the record-level rules).
_QUOTE_CHARS = set("\"'‘’“”`")
_P_LINE_ALLOWED_START_EXTRA = set("(")  # allowed leading non-alnum/non-quote char

# Rule (b) gate: only run langdetect on heading/button lines that already
# show some non-English signal, to avoid misfiring on short plain-English
# marketing copy. A line qualifies if it has >= 4 words AND either contains
# a non-ASCII letter (accents etc.) or contains >= 2 words from this small
# list of common Romance/Germanic/Dutch function words.
_LANG_CHECK_MIN_WORDS = 4
_LANG_CHECK_MIN_FUNCTION_WORDS = 2
_FUNCTION_WORDS = {
    "de", "la", "el", "los", "las", "y", "que", "para", "con",
    "und", "der", "die", "das", "ist", "nicht",
    "les", "des", "une", "pour", "vous", "il",
    "per", "che", "di", "non",
    "het", "een", "voor",
}

# Rule (c): a record is dropped outright (instead of trimmed) if (a)+(b)
# remove more than this many lines, OR more than this fraction of its lines.
_MAX_LINES_LOST_ABS = 5
_MAX_LINES_LOST_FRACTION = 0.25

# Ordered list of drop-reason codes; order matches the order rules are
# checked in (first failing rule wins), which is also the order they were
# specified in.
RULE_ORDER = [
    "parse_fail",
    "line_filter_too_many_removed",
    "hero_not_in_first3",
    "h1_count",
    "h1_words",
    "min_buttons",
    "p_count",
    "h2_min",
    "h2_ratio",
    "max_run",
    "total_words",
    "no_h3h4_or_cta",
    "line_too_long",
    "button_repeat",
    "numeric_lines",
    "banned_phrase",
]

RULE_DESCRIPTIONS = {
    "parse_fail": "Rule 1: a line failed to parse as <tag>...</tag>",
    "line_filter_too_many_removed": (
        "Rule (c): lost more than 5 lines, or more than 25% of its lines, "
        "to per-line rules (a) bad <p> starts + (b) non-English heading/button lines"
    ),
    "hero_not_in_first3": "Rule 2: h1 present but not within the first 3 elements",
    "h1_count": "Rule 3: h1 count != 1",
    "h1_words": "Rule 3: h1 word count not in [2,16]",
    "min_buttons": "Rule 3: button count < 1",
    "p_count": "Rule 3: p count not in [4,35]",
    "h2_min": "Rule 3: h2 count < 2",
    "h2_ratio": "Rule 3: h2 count / element count > 0.45",
    "max_run": "Rule 3: longest consecutive same-tag run > 8",
    "total_words": "Rule 3: total words not in [120,420]",
    "no_h3h4_or_cta": "Rule 3: no h3/h4 present and button count < 2",
    "line_too_long": "Rule 3: a line has > 90 words",
    "button_repeat": "Rule 3: a button text (case-folded) repeated > 2 times",
    "numeric_lines": "Rule 3: >= 4 pure number/price/date lines",
    "banned_phrase": (
        'Rule 3: a line contains "cookie" / "javascript" / '
        '"subscribe to our newsletter" (case-insensitive)'
    ),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def word_count(text):
    return len(text.split())


def _is_numeric_token(tok):
    t = tok.strip(",.;:")
    if not t:
        return False
    low = t.lower()
    if low in MONTHS or low in ("am", "pm"):
        return True
    if _ORDINAL_RE.match(low):
        return True
    if _DATE_SEP_RE.match(low):
        return True
    if _NUMERIC_RE.match(low):
        return True
    return False


def is_number_price_date(text):
    """True if `text` (a line's inner content) is *purely* a number, price,
    date, or similar short numeric callout (e.g. "$99", "2024", "Jan 1,
    2025", "50,000+", "24/7"), with no other prose."""
    t = text.strip()
    if not t or not any(ch.isdigit() for ch in t):
        return False
    tokens = t.split()
    if len(tokens) > 6:
        return False
    return all(_is_numeric_token(tok) for tok in tokens)


def should_drop_p_line(text):
    """Per-line rule (a): drop a <p> line's text only if it

      - starts with a lowercase letter, or
      - starts with a character that is not a letter, digit, quote mark, or
        opening parenthesis.
    """
    t = text.strip()
    if not t:
        return False
    first = t[0]

    if first.islower():
        return True

    if not (first.isalnum() or first in _QUOTE_CHARS or first in _P_LINE_ALLOWED_START_EXTRA):
        return True

    return False


def _has_non_ascii_letter(text):
    return any(ch.isalpha() and ord(ch) > 127 for ch in text)


def _count_function_words(text):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return sum(1 for tok in tokens if tok in _FUNCTION_WORDS)


def _passes_lang_check_gate(text):
    """Rule (b) gate: only lines with >= 4 words AND (a non-ASCII letter OR
    >= 2 function words from the list) are eligible for langdetect at all."""
    if word_count(text) < _LANG_CHECK_MIN_WORDS:
        return False
    if _has_non_ascii_letter(text):
        return True
    if _count_function_words(text) >= _LANG_CHECK_MIN_FUNCTION_WORDS:
        return True
    return False


def is_confidently_non_english(text):
    """Per-line rule (b): for a heading/button line that passes the
    vocabulary gate (`_passes_lang_check_gate`), drop it if langdetect is
    available and confident (probability > 0.9) that the top-detected
    language is not English. Lines that don't pass the gate are never run
    through langdetect and are always kept."""
    if detect_langs is None:
        return False
    if not _passes_lang_check_gate(text):
        return False
    try:
        results = detect_langs(text)
    except LangDetectException:
        return False
    except Exception:
        return False
    if not results:
        return False
    top = results[0]
    return top.lang != "en" and top.prob > 0.9


def get_domain(url):
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        netloc = ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def percentile(sorted_vals, pct):
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def parse_record(raw_target):
    """Parse `target` into a list of (tag, text) tuples, or None if any
    line fails to match `^<(h1|h2|h3|h4|p|button)>(.*)</\\1>$`."""
    lines = raw_target.split("\n")
    out = []
    for ln in lines:
        m = TAG_RE.match(ln)
        if not m:
            return None
        out.append((m.group(1), m.group(2)))
    return out


def apply_line_filters(elements):
    """Per-line rules (a)+(b)+(c), run before the record-level rules.

    Drops individual <p> lines that fail `should_drop_p_line` (rule a) and
    individual heading/button lines that fail `is_confidently_non_english`
    (rule b). If the total removed exceeds `_MAX_LINES_LOST_ABS` lines, or
    exceeds `_MAX_LINES_LOST_FRACTION` of the record's original line count,
    the whole record is dropped instead (rule c).

    Returns (filtered_elements, removed_by_a, removed_by_b, drop_reason).
    """
    kept = []
    removed_a = 0
    removed_b = 0
    for tag, text in elements:
        if tag == "p" and should_drop_p_line(text):
            removed_a += 1
            continue
        if tag in ("h1", "h2", "h3", "h4", "button") and is_confidently_non_english(text):
            removed_b += 1
            continue
        kept.append((tag, text))

    total_removed = removed_a + removed_b
    total_lines = len(elements)
    if total_removed > _MAX_LINES_LOST_ABS or (
        total_lines > 0 and total_removed > _MAX_LINES_LOST_FRACTION * total_lines
    ):
        return kept, removed_a, removed_b, "line_filter_too_many_removed"
    return kept, removed_a, removed_b, None


def normalize_hero(elements):
    """Move a single h1 to position 0 if it's within the first 3 elements.

    Returns (new_elements, drop_reason). drop_reason is None unless the
    (single) h1 exists but sits beyond index 2, in which case the record
    must be dropped. When h1 count != 1, elements are returned unchanged
    (the exactly-one-h1 rule in `check_quality` will drop it)."""
    h1_positions = [i for i, (t, _) in enumerate(elements) if t == "h1"]
    if len(h1_positions) != 1:
        return elements, None
    idx = h1_positions[0]
    if idx >= 3:
        return elements, "hero_not_in_first3"
    if idx == 0:
        return elements, None
    new_elements = [elements[idx]] + elements[:idx] + elements[idx + 1:]
    return new_elements, None


def longest_run(tags):
    if not tags:
        return 0
    best = cur = 1
    for i in range(1, len(tags)):
        if tags[i] == tags[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def check_quality(elements):
    """Check the rule-3 battery, in the specified order. Returns the first
    failing rule's code, or None if the record passes every rule."""
    tags = [t for t, _ in elements]
    texts = [x for _, x in elements]
    n = len(elements)

    if tags.count("h1") != 1:
        return "h1_count"
    h1_words = word_count(texts[tags.index("h1")])
    if not (2 <= h1_words <= 16):
        return "h1_words"

    button_count = tags.count("button")
    if button_count < 1:
        return "min_buttons"

    p_count = tags.count("p")
    if not (4 <= p_count <= 35):
        return "p_count"

    h2_count = tags.count("h2")
    if h2_count < 2:
        return "h2_min"
    if n > 0 and (h2_count / n) > 0.45:
        return "h2_ratio"

    if longest_run(tags) > 8:
        return "max_run"

    total_words = sum(word_count(t) for t in texts)
    if not (120 <= total_words <= 420):
        return "total_words"

    h3h4_count = tags.count("h3") + tags.count("h4")
    if not (h3h4_count >= 1 or button_count >= 2):
        return "no_h3h4_or_cta"

    if any(word_count(t) > 90 for t in texts):
        return "line_too_long"

    button_texts = Counter(
        t.strip().lower() for tag, t in elements if tag == "button"
    )
    if any(c > 2 for c in button_texts.values()):
        return "button_repeat"

    numeric_lines = sum(1 for t in texts if is_number_price_date(t))
    if numeric_lines >= 4:
        return "numeric_lines"

    lowered = [t.lower() for t in texts]
    if any(any(p in t for p in BANNED_PHRASES) for t in lowered):
        return "banned_phrase"

    return None


def rebuild_record(r, elements):
    new_target = "\n".join(f"<{tag}>{text}</{tag}>" for tag, text in elements)
    new_rec = dict(r)
    new_rec["target"] = new_target
    messages = [dict(m) for m in r["messages"]]
    messages[2] = dict(messages[2])
    messages[2]["content"] = new_target
    new_rec["messages"] = messages
    return new_rec


def curate_records(records):
    """Apply rules 1, 1.5 (a/b/c), 2-4 to a list of raw records.

    Returns (kept_records, drop_counter, total, line_filter_stats) where
    line_filter_stats is a dict with the number of lines removed by rule
    (a), the number removed by rule (b), and the number of records that
    were kept but had >=1 line trimmed by (a)/(b)."""
    kept = []
    drops = Counter()
    removed_a_total = 0
    removed_b_total = 0
    records_trimmed = 0
    for r in records:
        parsed = parse_record(r["target"])
        if parsed is None:
            drops["parse_fail"] += 1
            continue

        filtered, removed_a, removed_b, line_reason = apply_line_filters(parsed)
        removed_a_total += removed_a
        removed_b_total += removed_b
        if line_reason:
            drops[line_reason] += 1
            continue
        if removed_a + removed_b > 0:
            records_trimmed += 1

        elements, hero_reason = normalize_hero(filtered)
        if hero_reason:
            drops[hero_reason] += 1
            continue
        reason = check_quality(elements)
        if reason:
            drops[reason] += 1
            continue
        kept.append(rebuild_record(r, elements))

    line_filter_stats = {
        "removed_a": removed_a_total,
        "removed_b": removed_b_total,
        "records_trimmed": records_trimmed,
    }
    return kept, drops, len(records), line_filter_stats


def resplit(curated_train, curated_val, curated_test, target_val, target_test, seed):
    """Rule 5: keep curated val/test as-is, top up from curated train
    (seed-shuffled, domain-disjoint from current val+test) until val/test
    reach their targets. Everything else stays train."""
    domains_in_val_test = {get_domain(r["url"]) for r in curated_val}
    domains_in_val_test |= {get_domain(r["url"]) for r in curated_test}

    rng = random.Random(seed)
    shuffled = list(curated_train)
    rng.shuffle(shuffled)

    need_test = max(0, target_test - len(curated_test))
    need_val = max(0, target_val - len(curated_val))

    moved_ids = set()
    moved_to_test = []
    moved_to_val = []

    for rec in shuffled:
        if need_test <= 0 and need_val <= 0:
            break
        d = get_domain(rec["url"])
        if d in domains_in_val_test:
            continue
        if need_test > 0:
            moved_to_test.append(rec)
            moved_ids.add(rec["id"])
            domains_in_val_test.add(d)
            need_test -= 1
        elif need_val > 0:
            moved_to_val.append(rec)
            moved_ids.add(rec["id"])
            domains_in_val_test.add(d)
            need_val -= 1

    final_train = [r for r in curated_train if r["id"] not in moved_ids]
    final_test = curated_test + moved_to_test
    final_val = curated_val + moved_to_val

    info = {
        "moved_to_test": len(moved_to_test),
        "moved_to_val": len(moved_to_val),
        "shortfall_test": need_test,
        "shortfall_val": need_val,
    }
    return final_train, final_val, final_test, info


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            recs.append(json.loads(line))
    return recs


def write_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def tag_stats(records):
    """Per-record element/word counts and per-tag counts for a split."""
    word_counts = []
    elem_counts = []
    tag_totals = Counter()
    for r in records:
        elements = parse_record(r["target"])
        tags = [t for t, _ in elements]
        texts = [x for _, x in elements]
        word_counts.append(sum(word_count(t) for t in texts))
        elem_counts.append(len(elements))
        for t in tags:
            tag_totals[t] += 1
    return word_counts, elem_counts, tag_totals


def pick_examples(records, first_n=20, suffixes=("5", "7")):
    """From the first `first_n` records, pick (in order) the first record
    whose id ends with each of `suffixes`."""
    examples = {}
    for r in records[:first_n]:
        for suf in suffixes:
            if suf not in examples and r["id"].endswith(suf):
                examples[suf] = r
    return examples


def build_stats_md(
    input_counts,
    drop_by_file,
    final_splits,
    resplit_info,
    curated_base_counts,
    line_filter_stats_by_file,
):
    final_train, final_val, final_test = final_splits
    lines = []
    lines.append("# Dataset curation stats\n")

    # --- counts per split and per source ---
    lines.append("## Counts per split and per source\n")
    lines.append("| split | source | count |")
    lines.append("|---|---|---|")
    split_names = [("train", final_train), ("val", final_val), ("test", final_test)]
    for name, recs in split_names:
        src_counts = Counter(r["source"] for r in recs)
        for src in sorted(src_counts):
            lines.append(f"| {name} | {src} | {src_counts[src]} |")
        lines.append(f"| **{name}** | **total** | **{len(recs)}** |")
    grand_total = len(final_train) + len(final_val) + len(final_test)
    lines.append(f"| **all** | **total** | **{grand_total}** |")
    lines.append("")

    # --- curation + re-split summary ---
    lines.append("## Curation and re-split summary\n")
    lines.append("| input file | input records | passed rules 1-3 | drop rate |")
    lines.append("|---|---|---|---|")
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total = input_counts[fname]
        kept = curated_base_counts[fname]
        rate = 100.0 * (total - kept) / total if total else 0.0
        lines.append(f"| {fname} | {total} | {kept} | {rate:.1f}% |")
    lines.append("")
    lines.append(
        f"Re-split (seed=42): moved **{resplit_info['moved_to_test']}** curated "
        f"train records into test and **{resplit_info['moved_to_val']}** into val "
        f"(domain-disjoint from val/test), to reach the targets. "
        f"Final: train={len(final_train)}, val={len(final_val)}, test={len(final_test)}."
    )
    if resplit_info["shortfall_test"] or resplit_info["shortfall_val"]:
        lines.append(
            f"\n**Warning:** ran out of eligible curated-train candidates before "
            f"reaching target counts (shortfall_test={resplit_info['shortfall_test']}, "
            f"shortfall_val={resplit_info['shortfall_val']})."
        )
    lines.append("")

    # --- per-line filter rules (a)+(b)+(c) ---
    lines.append("## Per-line filter rules (a)+(b)+(c)\n")
    lines.append(
        "Applied before the record-level rules: (a) drops a `<p>` line only "
        "if it starts lowercase or starts with a char that isn't a "
        "letter/digit/quote/opening-parenthesis; (b) drops a heading/button "
        "line only if it has >= 4 words AND (a non-ASCII letter OR >= 2 "
        "Romance/Germanic/Dutch function words) AND langdetect is confident "
        "(prob > 0.9) it's non-English; (c) drops the whole record if "
        "(a)+(b) removed more than 5 lines, or more than 25% of its lines.\n"
    )
    lines.append("| input file | (a) `<p>` lines removed | (b) non-English heading/button lines removed | records trimmed (kept) | records dropped by (c) |")
    lines.append("|---|---|---|---|---|")
    total_a = total_b = total_trimmed = total_c = 0
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        lfs = line_filter_stats_by_file[fname]
        c_dropped = drop_by_file[fname].get("line_filter_too_many_removed", 0)
        lines.append(
            f"| {fname} | {lfs['removed_a']} | {lfs['removed_b']} | "
            f"{lfs['records_trimmed']} | {c_dropped} |"
        )
        total_a += lfs["removed_a"]
        total_b += lfs["removed_b"]
        total_trimmed += lfs["records_trimmed"]
        total_c += c_dropped
    lines.append(f"| **total** | **{total_a}** | **{total_b}** | **{total_trimmed}** | **{total_c}** |")
    lines.append("")

    # --- drop histogram ---
    lines.append("## Drop histogram (first failing rule per record)\n")
    lines.append("| rule | " + " | ".join(f"{f}" for f in ("train.jsonl", "val.jsonl", "test.jsonl")) + " | total | description |")
    lines.append("|---|---|---|---|---|---|")
    combined = Counter()
    for fname in ("train.jsonl", "val.jsonl", "test.jsonl"):
        combined.update(drop_by_file[fname])
    for code in RULE_ORDER:
        counts = [drop_by_file[f].get(code, 0) for f in ("train.jsonl", "val.jsonl", "test.jsonl")]
        total = combined.get(code, 0)
        if total == 0 and all(c == 0 for c in counts):
            continue
        lines.append(
            f"| {code} | {counts[0]} | {counts[1]} | {counts[2]} | {total} | {RULE_DESCRIPTIONS[code]} |"
        )
    total_dropped = sum(combined.values())
    lines.append(f"| **total dropped** | | | | **{total_dropped}** | |")
    lines.append("")

    # --- percentiles on curated train ---
    word_counts, elem_counts, tag_totals = tag_stats(final_train)
    word_counts_sorted = sorted(word_counts)
    elem_counts_sorted = sorted(elem_counts)
    lines.append("## Curated train: words / elements per target (percentiles)\n")
    lines.append("| metric | p10 | p50 | p90 |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| words | {:.1f} | {:.1f} | {:.1f} |".format(
            percentile(word_counts_sorted, 10),
            percentile(word_counts_sorted, 50),
            percentile(word_counts_sorted, 90),
        )
    )
    lines.append(
        "| elements | {:.1f} | {:.1f} | {:.1f} |".format(
            percentile(elem_counts_sorted, 10),
            percentile(elem_counts_sorted, 50),
            percentile(elem_counts_sorted, 90),
        )
    )
    lines.append("")

    # --- mean tag counts ---
    lines.append("## Curated train: mean count per tag per record\n")
    lines.append("| tag | mean count |")
    lines.append("|---|---|")
    n_train = len(final_train) or 1
    for t in TAGS:
        lines.append(f"| {t} | {tag_totals.get(t, 0) / n_train:.3f} |")
    lines.append("")

    # --- example targets ---
    lines.append(
        "## Example curated train targets (ids ending in 5 and 7, among the first 20)\n"
    )
    examples = pick_examples(final_train, first_n=20, suffixes=("5", "7"))
    for suf in ("5", "7"):
        rec = examples.get(suf)
        lines.append(f"### id ending in '{suf}'\n")
        if rec is None:
            lines.append("_No record among the first 20 curated train records has an id ending in "
                          f"'{suf}'._\n")
            continue
        lines.append(f"id: `{rec['id']}`  name: {rec['name']}  url: {rec['url']}\n")
        lines.append("```")
        lines.append(rec["target"])
        lines.append("```")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curate and re-split the copywriting landing-page dataset."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                         help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output-dir", default=None,
                         help="Output directory (default: <input-dir>/../dataset_curated)")
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--val-file", default="val.jsonl")
    parser.add_argument("--test-file", default="test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-val", type=int, default=100)
    parser.add_argument("--target-test", type=int, default=120)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "dataset_curated"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        "train.jsonl": input_dir / args.train_file,
        "val.jsonl": input_dir / args.val_file,
        "test.jsonl": input_dir / args.test_file,
    }

    raw = {}
    for key, path in file_map.items():
        if not path.exists():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            sys.exit(1)
        raw[key] = load_jsonl(path)

    input_counts = {k: len(v) for k, v in raw.items()}

    curated = {}
    drop_by_file = {}
    line_filter_stats_by_file = {}
    for key in ("train.jsonl", "val.jsonl", "test.jsonl"):
        kept, drops, total, line_filter_stats = curate_records(raw[key])
        curated[key] = kept
        drop_by_file[key] = drops
        line_filter_stats_by_file[key] = line_filter_stats

    curated_base_counts = {k: len(v) for k, v in curated.items()}

    final_train, final_val, final_test, resplit_info = resplit(
        curated["train.jsonl"], curated["val.jsonl"], curated["test.jsonl"],
        target_val=args.target_val, target_test=args.target_test, seed=args.seed,
    )

    write_jsonl(output_dir / "train.jsonl", final_train)
    write_jsonl(output_dir / "val.jsonl", final_val)
    write_jsonl(output_dir / "test.jsonl", final_test)

    stats_md = build_stats_md(
        input_counts=input_counts,
        drop_by_file=drop_by_file,
        final_splits=(final_train, final_val, final_test),
        resplit_info=resplit_info,
        curated_base_counts=curated_base_counts,
        line_filter_stats_by_file=line_filter_stats_by_file,
    )
    (output_dir / "stats.md").write_text(stats_md, encoding="utf-8")

    # ---- console report ----
    print("=" * 72)
    print("CURATION REPORT")
    print("=" * 72)
    for key in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total = input_counts[key]
        kept = curated_base_counts[key]
        lfs = line_filter_stats_by_file[key]
        print(f"\n[{key}] input={total}  kept(rules 1-3)={kept}  dropped={total - kept}")
        print(
            f"  per-line rules (a)+(b): removed {lfs['removed_a']} <p> lines (a), "
            f"{lfs['removed_b']} non-English heading/button lines (b); "
            f"{lfs['records_trimmed']} surviving records had lines trimmed"
        )
        print("  drop reasons (first failing rule):")
        for code in RULE_ORDER:
            c = drop_by_file[key].get(code, 0)
            if c:
                print(f"    {code:<28} {c:>5}   ({RULE_DESCRIPTIONS[code]})")

    total_removed_a = sum(s["removed_a"] for s in line_filter_stats_by_file.values())
    total_removed_b = sum(s["removed_b"] for s in line_filter_stats_by_file.values())
    total_dropped_c = sum(drop_by_file[k].get("line_filter_too_many_removed", 0) for k in drop_by_file)
    print(
        f"\nPer-line rules (a)+(b) totals across all input files: "
        f"{total_removed_a} <p> lines removed (a), {total_removed_b} heading/button "
        f"lines removed (b), {total_removed_a + total_removed_b} lines removed in total; "
        f"{total_dropped_c} records dropped by rule (c) (>5 lines or >25% of lines lost)."
    )

    print("\nRe-split (seed={}):".format(args.seed))
    print(f"  moved_to_test={resplit_info['moved_to_test']}  moved_to_val={resplit_info['moved_to_val']}")
    if resplit_info["shortfall_test"] or resplit_info["shortfall_val"]:
        print(f"  WARNING shortfall_test={resplit_info['shortfall_test']} shortfall_val={resplit_info['shortfall_val']}")

    print("\nFinal split sizes:")
    print(f"  train={len(final_train)}  val={len(final_val)}  test={len(final_test)}")
    print(f"\nWrote output to: {output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
