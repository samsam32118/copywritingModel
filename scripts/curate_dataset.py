#!/usr/bin/env python3
"""curate_dataset.py

Curates the landing-page copywriting dataset (train/val/test JSONL) into a
quality-filtered, re-split copy.

Pipeline (applied identically to train.jsonl, val.jsonl, test.jsonl):
  1. Parse every line of `target` as a `<tag>text</tag>` element (h1/h2/h3/
     h4/p/button). Any record with a line that fails to parse is dropped.
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

# Ordered list of drop-reason codes; order matches the order rules are
# checked in (first failing rule wins), which is also the order they were
# specified in.
RULE_ORDER = [
    "parse_fail",
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
    """Apply rules 1-4 to a list of raw records.

    Returns (kept_records, drop_counter, total)."""
    kept = []
    drops = Counter()
    for r in records:
        parsed = parse_record(r["target"])
        if parsed is None:
            drops["parse_fail"] += 1
            continue
        elements, hero_reason = normalize_hero(parsed)
        if hero_reason:
            drops[hero_reason] += 1
            continue
        reason = check_quality(elements)
        if reason:
            drops[reason] += 1
            continue
        kept.append(rebuild_record(r, elements))
    return kept, drops, len(records)


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
    for key in ("train.jsonl", "val.jsonl", "test.jsonl"):
        kept, drops, total = curate_records(raw[key])
        curated[key] = kept
        drop_by_file[key] = drops

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
    )
    (output_dir / "stats.md").write_text(stats_md, encoding="utf-8")

    # ---- console report ----
    print("=" * 72)
    print("CURATION REPORT")
    print("=" * 72)
    for key in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total = input_counts[key]
        kept = curated_base_counts[key]
        print(f"\n[{key}] input={total}  kept(rules 1-3)={kept}  dropped={total - kept}")
        print("  drop reasons (first failing rule):")
        for code in RULE_ORDER:
            c = drop_by_file[key].get(code, 0)
            if c:
                print(f"    {code:<20} {c:>5}   ({RULE_DESCRIPTIONS[code]})")

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
