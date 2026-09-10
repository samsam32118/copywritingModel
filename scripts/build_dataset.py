#!/usr/bin/env python3
"""Build train/val/test fine-tuning datasets from scraped landing pages.

CLI:
    python3 build_dataset.py --inputs raw_yc.jsonl [raw_gallery.jsonl ...] --out-dir dataset/

Reads JSONL records produced by scrape_pages.py, applies a quality filter,
renders a (brief, target) pair per surviving page, dedupes, splits by
registrable domain into train/val/test, and writes:
    <out-dir>/train.jsonl
    <out-dir>/val.jsonl
    <out-dir>/test.jsonl
    <out-dir>/stats.md
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

from prompt import SYSTEM_PROMPT

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

FORBIDDEN_PHRASES = [
    "lorem ipsum",
    "page not found",
    "404",
    "access denied",
    "enable javascript",
    "checking your browser",
]
BAD_TITLE_RE = re.compile(r"(?i)404|not found|error|just a moment|attention required|access denied")

TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "co.jp", "ne.jp", "or.jp", "co.nz",
    "co.za", "com.au", "net.au", "org.au", "co.in", "com.br", "com.mx",
    "co.id", "com.sg", "com.tr", "com.tw", "co.kr", "com.hk", "com.cn",
    "co.il", "com.ar", "co.th",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def normalize_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def word_count(text):
    return len(text.split()) if text else 0


def registrable_domain(url):
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return (url or "").lower()
    netloc = netloc.split("@")[-1]
    netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    labels = [l for l in netloc.split(".") if l]
    if len(labels) <= 2:
        return netloc
    last_two = ".".join(labels[-2:])
    if last_two in TWO_PART_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def trim_at_sentence(text, limit=500):
    text = normalize_ws(text)
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    matches = list(re.finditer(r"[.!?](?=\s|$)", truncated))
    if matches:
        cut = matches[-1].end()
        result = truncated[:cut].strip()
        if result:
            return result
    sp = truncated.rfind(" ")
    if sp > 40:  # avoid chopping to almost nothing
        return truncated[:sp].strip() + "…"
    return truncated.strip() + "…"


def clean_title_suffix(title):
    """Strip a trailing ' | Site Name' suffix, keep the leading page title."""
    t = normalize_ws(title)
    if "|" in t:
        first = t.split("|", 1)[0].strip()
        if len(first) >= 2:
            return first
    return t


def clean_subindustry(subindustry, industry):
    if not subindustry:
        return ""
    if "->" in subindustry:
        part = subindustry.split("->")[-1].strip()
    else:
        part = subindustry.strip()
    if industry and part.lower() == industry.strip().lower():
        return ""
    return part


def normalize_for_match(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def remove_leaked_sentences(description, target_text, min_overlap_words=12):
    """Drop any sentence from `description` that shares a >=min_overlap_words
    verbatim run with `target_text` (case/whitespace-insensitive)."""
    if not description:
        return description
    target_norm = normalize_for_match(target_text)
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    kept = []
    for sent in sentences:
        words = normalize_for_match(sent).split()
        leaked = False
        if len(words) >= min_overlap_words:
            for i in range(0, len(words) - min_overlap_words + 1):
                window = " ".join(words[i:i + min_overlap_words])
                if window and window in target_norm:
                    leaked = True
                    break
        if not leaked:
            kept.append(sent)
    return " ".join(kept).strip()


def percentile(values, p):
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# --------------------------------------------------------------------------
# Quality filter (operates on the full, pre-truncation element list)
# --------------------------------------------------------------------------

def apply_quality_filter(rec):
    """Returns (elements_or_None, drop_reason_or_None).

    On success, elements is the (possibly h1-demoted) element list -- NOT
    yet truncated to the word/element caps.
    """
    lang = rec.get("lang")
    html_lang = (rec.get("html_lang") or "").lower()
    lang_ok = (lang == "en") or (lang is None and html_lang.startswith("en"))
    if not lang_ok:
        return None, "non_english"

    elements = [dict(e) for e in (rec.get("elements") or [])]

    h1_idx = [i for i, e in enumerate(elements) if e["tag"] == "h1"]
    if not h1_idx:
        return None, "no_h1"
    first_h1 = h1_idx[0]
    wc = word_count(elements[first_h1]["text"])
    if not (2 <= wc <= 20):
        return None, "h1_wordcount"
    for i in h1_idx[1:]:
        elements[i]["tag"] = "h2"

    if sum(1 for e in elements if e["tag"] == "h2") < 2:
        return None, "too_few_h2"
    if sum(1 for e in elements if e["tag"] == "p") < 3:
        return None, "too_few_p"
    if sum(1 for e in elements if e["tag"] == "button") < 1:
        return None, "too_few_button"

    total_words = sum(word_count(e["text"]) for e in elements)
    if total_words < 60:
        return None, "too_few_words"

    combined = " ".join(e["text"] for e in elements)
    combined_lower = combined.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in combined_lower:
            return None, f"forbidden_phrase:{phrase}"

    title = rec.get("title") or ""
    if BAD_TITLE_RE.search(title):
        return None, "bad_title"

    if combined:
        non_ascii = sum(1 for ch in combined if ord(ch) > 127)
        if non_ascii / len(combined) > 0.30:
            return None, "non_ascii_heavy"

    return elements, None


def truncate_elements(elements, max_words=420, max_elements=60):
    truncated = []
    word_total = 0
    for e in elements:
        wc = word_count(e["text"])
        if len(truncated) >= max_elements:
            break
        if truncated and (word_total + wc) > max_words:
            break
        truncated.append(e)
        word_total += wc
    if not any(e["tag"] == "button" for e in truncated):
        first_button = next((e for e in elements if e["tag"] == "button"), None)
        if first_button is not None:
            truncated.append(first_button)
    return truncated


def render_target(elements):
    return "\n".join(f"<{e['tag']}>{e['text']}</{e['tag']}>" for e in elements)


# --------------------------------------------------------------------------
# Brief construction
# --------------------------------------------------------------------------

def build_brief_yc(rec, target_text, desc_limit):
    yc = rec.get("yc") or {}
    name = rec.get("name") or ""
    one_liner = normalize_ws(yc.get("one_liner") or "")
    description = trim_at_sentence(yc.get("long_description") or "", desc_limit)
    description = remove_leaked_sentences(description, target_text)
    industry = normalize_ws(yc.get("industry") or "")
    subindustry = clean_subindustry(yc.get("subindustry") or "", industry)
    tags = [t for t in (yc.get("tags") or []) if t]

    lines = [f"Product: {name}"]
    if one_liner:
        lines.append(f"One-liner: {one_liner}")
    if description:
        lines.append(f"Description: {description}")
    if industry:
        if subindustry:
            lines.append(f"Industry: {industry} / {subindustry}")
        else:
            lines.append(f"Industry: {industry}")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    return "\n".join(lines), name


def build_brief_gallery(rec, target_text, desc_limit, min_desc_chars):
    raw_title = rec.get("og_title") or rec.get("title") or ""
    name = clean_title_suffix(raw_title)
    if not name:
        return None, None, "gallery_no_name"

    raw_desc = normalize_ws(rec.get("meta_description") or rec.get("og_description") or "")
    if len(raw_desc) < min_desc_chars:
        return None, None, "gallery_short_description"

    description = trim_at_sentence(raw_desc, desc_limit)
    description = remove_leaked_sentences(description, target_text)

    lines = [f"Product: {name}"]
    if description:
        lines.append(f"Description: {description}")
    return "\n".join(lines), name, None


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def process_all(input_paths, args):
    drop_reasons = Counter()
    per_source_read = Counter()
    kept = []          # list of dict output records (without id yet)
    kept_stats = []    # parallel list of (n_words, n_elements, tag_counter)
    seen_domains = set()
    seen_target_keys = set()
    source_counters = defaultdict(int)

    for path in input_paths:
        rows = load_jsonl(path)
        for rec in rows:
            source = rec.get("source") or "unknown"
            per_source_read[source] += 1

            elements, reason = apply_quality_filter(rec)
            if reason:
                drop_reasons[reason] += 1
                continue

            truncated = truncate_elements(elements, args.max_target_words, args.max_elements)
            target_text = render_target(truncated)

            is_yc = rec.get("yc") is not None
            if is_yc:
                brief, name = build_brief_yc(rec, target_text, args.max_brief_desc_chars)
                brief_reason = None
            else:
                brief, name, brief_reason = build_brief_gallery(
                    rec, target_text, args.max_brief_desc_chars, args.min_gallery_desc_chars
                )
            if brief_reason:
                drop_reasons[brief_reason] += 1
                continue

            url = rec.get("url") or ""
            domain = registrable_domain(rec.get("final_url") or url)
            if not domain:
                drop_reasons["no_domain"] += 1
                continue
            if domain in seen_domains:
                drop_reasons["dup_domain"] += 1
                continue

            target_key = normalize_for_match(target_text)[:300]
            if target_key in seen_target_keys:
                drop_reasons["dup_near_target"] += 1
                continue

            seen_domains.add(domain)
            seen_target_keys.add(target_key)

            source_counters[source] += 1
            rec_id = f"{source}-{source_counters[source]:05d}"

            out_rec = {
                "id": rec_id,
                "source": source,
                "name": name,
                "url": url,
                "brief": brief,
                "target": target_text,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": brief},
                    {"role": "assistant", "content": target_text},
                ],
                "_domain": domain,
            }
            kept.append(out_rec)

            tag_counter = Counter(e["tag"] for e in truncated)
            kept_stats.append((word_count(target_text.replace("\n", " ")), len(truncated), tag_counter))

    return kept, kept_stats, drop_reasons, per_source_read


def split_by_domain(kept, seed, min_test, min_val):
    groups = defaultdict(list)
    for rec in kept:
        groups[rec["_domain"]].append(rec)
    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)
    flat = [r for g in group_list for r in g]

    total_n = len(flat)
    n_test = min(max(round(total_n * 0.05), min_test), total_n)
    n_val = min(max(round(total_n * 0.05), min_val), total_n - n_test)
    n_train = total_n - n_test - n_val

    test = flat[:n_test]
    val = flat[n_test:n_test + n_val]
    train = flat[n_test + n_val:]
    assert len(train) == n_train
    return train, val, test


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def write_stats(path, input_paths, per_source_read, drop_reasons, kept_stats_by_id,
                 train, val, test, examples):
    all_splits = {"train": train, "val": val, "test": test}
    sources = sorted(per_source_read.keys())

    lines = []
    lines.append("# Dataset build stats\n")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")

    lines.append("## Inputs\n")
    for p in input_paths:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("Raw records read per source:\n")
    lines.append("| source | records read |")
    lines.append("|---|---|")
    for s in sources:
        lines.append(f"| {s} | {per_source_read[s]} |")
    lines.append("")

    lines.append("## Counts per split x source\n")
    lines.append("| split | " + " | ".join(sources) + " | total |")
    lines.append("|---|" + "---|" * (len(sources) + 1))
    grand_total = 0
    for split_name, recs in all_splits.items():
        counts = Counter(r["source"] for r in recs)
        row = [split_name] + [str(counts.get(s, 0)) for s in sources] + [str(len(recs))]
        lines.append("| " + " | ".join(row) + " |")
        grand_total += len(recs)
    total_counts = Counter(r["source"] for split in all_splits.values() for r in split)
    lines.append(
        "| **total** | "
        + " | ".join(str(total_counts.get(s, 0)) for s in sources)
        + f" | {grand_total} |"
    )
    lines.append("")

    lines.append("## Drop reasons\n")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for reason, count in drop_reasons.most_common():
        lines.append(f"| {reason} | {count} |")
    if not drop_reasons:
        lines.append("| (none) | 0 |")
    lines.append("")

    all_words = [w for (w, n, tc) in kept_stats_by_id]
    all_nelem = [n for (w, n, tc) in kept_stats_by_id]
    lines.append(f"## Target length distribution (kept examples, n={len(kept_stats_by_id)})\n")
    lines.append("| metric | p10 | p50 | p90 |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| words per target | {percentile(all_words, 0.10):.0f} | "
        f"{percentile(all_words, 0.50):.0f} | {percentile(all_words, 0.90):.0f} |"
    )
    lines.append(
        f"| elements per target | {percentile(all_nelem, 0.10):.0f} | "
        f"{percentile(all_nelem, 0.50):.0f} | {percentile(all_nelem, 0.90):.0f} |"
    )
    lines.append("")

    tag_totals = Counter()
    for (w, n, tc) in kept_stats_by_id:
        tag_totals.update(tc)
    n_examples = max(len(kept_stats_by_id), 1)
    lines.append(f"## Tag counts (kept examples, n={len(kept_stats_by_id)})\n")
    lines.append("| tag | total count | avg per example |")
    lines.append("|---|---|---|")
    for tag in ["h1", "h2", "h3", "h4", "p", "button"]:
        total = tag_totals.get(tag, 0)
        lines.append(f"| {tag} | {total} | {total / n_examples:.2f} |")
    lines.append("")

    lines.append("## Example records\n")
    for label, rec in examples:
        lines.append(f"### {label}\n")
        lines.append(f"id: `{rec['id']}`  source: `{rec['source']}`  url: {rec['url']}\n")
        lines.append("Brief:")
        lines.append("```")
        lines.append(rec["brief"])
        lines.append("```")
        tgt_lines = rec["target"].split("\n")
        preview = tgt_lines[:12]
        lines.append("Target (preview):")
        lines.append("```")
        lines.extend(preview)
        if len(tgt_lines) > len(preview):
            lines.append(f"... ({len(tgt_lines) - len(preview)} more lines)")
        lines.append("```")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="Input raw_*.jsonl files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-test", type=int, default=120)
    ap.add_argument("--min-val", type=int, default=80)
    ap.add_argument("--max-target-words", type=int, default=420)
    ap.add_argument("--max-elements", type=int, default=60)
    ap.add_argument("--max-brief-desc-chars", type=int, default=500)
    ap.add_argument("--min-gallery-desc-chars", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    kept, kept_stats, drop_reasons, per_source_read = process_all(args.inputs, args)
    print(f"kept={len(kept)} dropped={sum(drop_reasons.values())}")

    train, val, test = split_by_domain(kept, args.seed, args.min_test, args.min_val)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    id_to_stats = {r["id"]: s for r, s in zip(kept, kept_stats)}
    kept_stats_ordered = [id_to_stats[r["id"]] for split in (train, val, test) for r in split]

    examples = []
    if train:
        examples.append(("Example 1 (train)", train[0]))
    if val:
        examples.append(("Example 2 (val)", val[0]))
    if test:
        examples.append(("Example 3 (test)", test[0]))

    write_stats(
        os.path.join(args.out_dir, "stats.md"),
        args.inputs, per_source_read, drop_reasons, kept_stats_ordered,
        train, val, test, examples,
    )
    print(f"wrote dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
