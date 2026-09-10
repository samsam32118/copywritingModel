#!/usr/bin/env python3
"""Score a predictions JSONL (from generate.py) against its references on
format validity, structural shape, and content quality/faithfulness.

Writes --out as JSON and a same-named .md table report.

Example:
    python scripts/evaluate.py --preds preds.jsonl --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter

import eval_common as ec
from rouge_score import rouge_scorer

# A small set of common CTA imperative verbs (lowercase, no punctuation).
IMPERATIVE_VERBS = {
    "get", "start", "try", "join", "sign", "buy", "download", "learn", "book",
    "request", "subscribe", "explore", "discover", "claim", "unlock", "contact",
    "schedule", "shop", "order", "upgrade", "register", "watch", "see", "view",
    "build", "create", "add", "grab", "reserve", "begin", "launch", "browse",
    "find", "save", "talk", "chat", "call", "apply", "connect", "activate",
}

MARKDOWN_MARKERS = ("```", "##", "**", "__")


def parse_args():
    p = argparse.ArgumentParser(
        description="Score generate.py predictions: format validity, structure, content.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preds", required=True, help="Predictions JSONL from generate.py (id, brief, reference, prediction, ...).")
    p.add_argument("--out", default="results.json", help="Output JSON path; a sibling .md table is written alongside it.")
    return p.parse_args()


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def distribution_stats(lst):
    if not lst:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0}
    return {"n": len(lst), "mean": mean(lst), "median": statistics.median(lst), "min": min(lst), "max": max(lst)}


def pct_in_range(lst, lo, hi):
    if not lst:
        return 0.0
    return sum(1 for x in lst if lo <= x <= hi) / len(lst)


def fmt_pct_or_na(x):
    return "n/a" if x is None else f"{x:.1%}"


def fmt_or_na(x, spec="{:.3f}"):
    return "n/a" if x is None else spec.format(x)


# ---------------------------------------------------------------------------
# a) Format validity
# ---------------------------------------------------------------------------


def format_checks(prediction: str, reference: str) -> dict:
    lines = ec.parse_lines(prediction)  # blank lines already dropped

    all_lines_valid = all(l.valid for l in lines) if lines else False
    tag_counts = Counter(l.tag for l in lines if l.valid)

    raws = [l.raw for l in lines]
    no_duplicate_lines = (len(raws) == len(set(raws))) if raws else False

    has_markdown = any(m in prediction for m in MARKDOWN_MARKERS)
    has_bullets = any(l.raw.startswith(("- ", "* ", "> ", "1. ")) for l in lines)
    no_extra_text = all_lines_valid and not has_markdown and not has_bullets

    pred_words = len(prediction.split())
    ref_words = len(reference.split())
    length_ok = (pred_words == 0) if ref_words == 0 else (0.4 * ref_words <= pred_words <= 1.8 * ref_words)

    checks = {
        "all_lines_valid": all_lines_valid,
        "exactly_one_h1": tag_counts.get("h1", 0) == 1,
        "h2_at_least_2": tag_counts.get("h2", 0) >= 2,
        "p_at_least_3": tag_counts.get("p", 0) >= 3,
        "button_at_least_1": tag_counts.get("button", 0) >= 1,
        "no_duplicate_lines": no_duplicate_lines,
        "no_extra_text": no_extra_text,
        "length_ok": length_ok,
    }
    checks["overall_valid"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# b) Structure
# ---------------------------------------------------------------------------


def compute_structure(records: list[dict]) -> dict:
    pred_counts_per_ex, ref_counts_per_ex = [], []
    pred_words_per_tag = {t: [] for t in ec.TAG_NAMES}
    ref_words_per_tag = {t: [] for t in ec.TAG_NAMES}
    h1_word_counts, button_word_counts, button_imperative_flags, first_is_h1_flags = [], [], [], []

    for r in records:
        pred_lines = ec.valid_tag_lines(r["prediction"])
        ref_lines = ec.valid_tag_lines(r["reference"])

        pred_counts_per_ex.append(Counter(l.tag for l in pred_lines))
        ref_counts_per_ex.append(Counter(l.tag for l in ref_lines))

        for l in pred_lines:
            pred_words_per_tag[l.tag].append(len(l.content.split()))
        for l in ref_lines:
            ref_words_per_tag[l.tag].append(len(l.content.split()))

        for l in pred_lines:
            if l.tag == "h1":
                h1_word_counts.append(len(l.content.split()))
            elif l.tag == "button":
                bwords = l.content.split()
                button_word_counts.append(len(bwords))
                first_word = re.sub(r"[^a-zA-Z]", "", bwords[0]).lower() if bwords else ""
                button_imperative_flags.append(first_word in IMPERATIVE_VERBS)

        all_lines = ec.parse_lines(r["prediction"])
        first_is_h1_flags.append(bool(all_lines) and all_lines[0].tag == "h1")

    def mean_counts(counts_list):
        return {t: mean([c.get(t, 0) for c in counts_list]) for t in ec.TAG_NAMES}

    def mean_proportions(counts_list):
        props = []
        for c in counts_list:
            total = sum(c.values())
            if total > 0:
                props.append([c.get(t, 0) / total for t in ec.TAG_NAMES])
        if not props:
            return [0.0] * len(ec.TAG_NAMES)
        return [sum(p[i] for p in props) / len(props) for i in range(len(ec.TAG_NAMES))]

    pred_mean_props = mean_proportions(pred_counts_per_ex)
    ref_mean_props = mean_proportions(ref_counts_per_ex)

    h1_stats = distribution_stats(h1_word_counts)
    h1_stats["pct_4_12_words"] = pct_in_range(h1_word_counts, 4, 12)
    button_stats = distribution_stats(button_word_counts)
    button_stats["pct_1_5_words"] = pct_in_range(button_word_counts, 1, 5)

    return {
        "mean_tag_count": {"pred": mean_counts(pred_counts_per_ex), "ref": mean_counts(ref_counts_per_ex)},
        "tag_js_divergence": ec.js_divergence(pred_mean_props, ref_mean_props),
        "mean_words_per_tag": {
            "pred": {t: mean(v) for t, v in pred_words_per_tag.items()},
            "ref": {t: mean(v) for t, v in ref_words_per_tag.items()},
        },
        "h1_word_count": h1_stats,
        "button_word_count": button_stats,
        "button_imperative_rate": mean(button_imperative_flags),
        "first_element_h1_rate": mean(first_is_h1_flags),
    }


# ---------------------------------------------------------------------------
# c) Content
# ---------------------------------------------------------------------------


def compute_content(records: list[dict]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    name_hits, name_parsed = 0, 0
    keyword_coverages = []
    distinct2s = []
    repetition_flags = []
    hallucinated_counts = []
    gen_token_counts = []

    for r in records:
        pred, ref, brief = r["prediction"], r["reference"], r["brief"]

        sc = scorer.score(ref, pred)
        r1.append(sc["rouge1"].fmeasure)
        r2.append(sc["rouge2"].fmeasure)
        rl.append(sc["rougeL"].fmeasure)

        name = ec.product_name(brief)
        if name:
            name_parsed += 1
            if name.lower() in pred.lower():
                name_hits += 1

        freq = Counter(ec.content_words(brief))
        top_kw = [w for w, _ in freq.most_common(15)]
        if top_kw:
            pred_lower = pred.lower()
            keyword_coverages.append(sum(1 for w in top_kw if w in pred_lower) / len(top_kw))

        # Content-only text (tag markup stripped) so tag names like <h1>/<h2>
        # don't contaminate word/number metrics with stray digits/letters.
        pred_content = ec.content_text(pred)

        toks = ec.words(pred_content)
        bigrams = ec.ngrams(toks, 2)
        if bigrams:
            distinct2s.append(len(set(bigrams)) / len(bigrams))

        fourgrams = ec.ngrams(toks, 4)
        rep_flag = any(c >= 3 for c in Counter(fourgrams).values()) if fourgrams else False
        repetition_flags.append(rep_flag)

        pred_nums = ec.NUMERIC_RE.findall(pred_content)
        brief_nums = set(ec.NUMERIC_RE.findall(brief))
        hallucinated_counts.append(sum(1 for tok in pred_nums if tok not in brief_nums))

        if "n_new_tokens" in r and r["n_new_tokens"] is not None:
            gen_token_counts.append(r["n_new_tokens"])

    if gen_token_counts:
        mean_gen_tokens, gen_tokens_source = mean(gen_token_counts), "n_new_tokens"
    else:
        mean_gen_tokens = mean([len(r["prediction"].split()) for r in records])
        gen_tokens_source = "prediction_word_count_fallback"

    return {
        "rouge1_f1": mean(r1),
        "rouge2_f1": mean(r2),
        "rougeL_f1": mean(rl),
        "product_mention_rate": (name_hits / name_parsed) if name_parsed else None,
        "product_name_parsed_count": name_parsed,
        "keyword_coverage": mean(keyword_coverages) if keyword_coverages else None,
        "distinct_2": mean(distinct2s) if distinct2s else None,
        "repetition_rate": mean(repetition_flags),
        "hallucinated_numbers_mean": mean(hallucinated_counts),
        "mean_generated_tokens": mean_gen_tokens,
        "mean_generated_tokens_source": gen_tokens_source,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(results: dict) -> str:
    s, f, st, c = results["summary"], results["format"], results["structure"], results["content"]
    lines = []
    lines.append("# Evaluation Results\n")
    lines.append(f"Predictions: `{results['preds_file']}`  \nn = {results['n']}\n")

    lines.append("## Summary (decision-relevant numbers)\n")
    lines.append(f"- **Format valid**: {s['overall_valid_rate']:.1%}")
    lines.append(f"- **ROUGE-L F1**: {s['rougeL_f1']:.3f}")
    lines.append(f"- **Product mention rate**: {fmt_pct_or_na(s['product_mention_rate'])}")
    lines.append(f"- **Brief keyword coverage**: {fmt_pct_or_na(s['keyword_coverage'])}")
    lines.append(f"- **Exactly-one-H1 rate**: {s['exactly_one_h1_rate']:.1%}")
    lines.append(f"- **Repetition rate**: {s['repetition_rate']:.1%}\n")

    lines.append("## Format validity (pass rate per check)\n")
    lines.append("| check | pass rate |")
    lines.append("|---|---|")
    for k, v in f.items():
        label = "**overall_valid**" if k == "overall_valid" else k
        lines.append(f"| {label} | {v:.1%} |")

    lines.append("\n## Structure\n")
    lines.append("| tag | mean count (pred) | mean count (ref) | mean words (pred) | mean words (ref) |")
    lines.append("|---|---|---|---|---|")
    for t in ec.TAG_NAMES:
        lines.append(
            f"| {t} | {st['mean_tag_count']['pred'][t]:.2f} | {st['mean_tag_count']['ref'][t]:.2f} "
            f"| {st['mean_words_per_tag']['pred'][t]:.2f} | {st['mean_words_per_tag']['ref'][t]:.2f} |"
        )
    lines.append(f"\n- Tag-proportion Jensen-Shannon divergence (pred vs ref): {st['tag_js_divergence']:.4f}")
    h1 = st["h1_word_count"]
    lines.append(f"- H1 word count: mean={h1['mean']:.1f}, median={h1['median']:.1f}, range=[{h1['min']},{h1['max']}], within 4-12 words={h1['pct_4_12_words']:.1%}")
    btn = st["button_word_count"]
    lines.append(f"- Button word count: mean={btn['mean']:.1f}, median={btn['median']:.1f}, within 1-5 words={btn['pct_1_5_words']:.1%}")
    lines.append(f"- Buttons starting with an imperative verb: {st['button_imperative_rate']:.1%}")
    lines.append(f"- First element is H1: {st['first_element_h1_rate']:.1%}")

    lines.append("\n## Content\n")
    lines.append(f"- ROUGE-1 / ROUGE-2 / ROUGE-L F1: {c['rouge1_f1']:.3f} / {c['rouge2_f1']:.3f} / {c['rougeL_f1']:.3f}")
    lines.append(f"- Product mention rate: {fmt_pct_or_na(c['product_mention_rate'])} (parsed from {c['product_name_parsed_count']}/{results['n']} briefs)")
    lines.append(f"- Brief keyword coverage (top-15 content words): {fmt_pct_or_na(c['keyword_coverage'])}")
    lines.append(f"- Distinct-2 ratio: {fmt_or_na(c['distinct_2'])}")
    lines.append(f"- Repetition rate (any 4-gram repeated >=3x): {c['repetition_rate']:.1%}")
    lines.append(f"- Hallucinated numeric tokens (mean per example): {c['hallucinated_numbers_mean']:.2f}")
    lines.append(f"- Mean generated tokens: {c['mean_generated_tokens']:.1f} (source: {c['mean_generated_tokens_source']})")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    records = ec.read_jsonl(args.preds)
    if not records:
        print(f"[evaluate] ERROR: no records found in {args.preds}", file=sys.stderr)
        sys.exit(1)

    format_per_record = [format_checks(r["prediction"], r["reference"]) for r in records]
    format_agg = {k: mean([c[k] for c in format_per_record]) for k in format_per_record[0]}

    structure = compute_structure(records)
    content = compute_content(records)

    summary = {
        "overall_valid_rate": format_agg["overall_valid"],
        "rougeL_f1": content["rougeL_f1"],
        "product_mention_rate": content["product_mention_rate"],
        "keyword_coverage": content["keyword_coverage"],
        "exactly_one_h1_rate": format_agg["exactly_one_h1"],
        "repetition_rate": content["repetition_rate"],
    }

    results = {
        "n": len(records),
        "preds_file": args.preds,
        "format": format_agg,
        "structure": structure,
        "content": content,
        "summary": summary,
    }

    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(render_markdown(results))

    print(f"[evaluate] n={len(records)} -> {args.out}, {md_path}", file=sys.stderr)
    print(
        "[evaluate] SUMMARY: "
        f"valid={summary['overall_valid_rate']:.1%} "
        f"rougeL={summary['rougeL_f1']:.3f} "
        f"product_mention={fmt_pct_or_na(summary['product_mention_rate'])} "
        f"keyword_coverage={fmt_pct_or_na(summary['keyword_coverage'])} "
        f"exactly_one_h1={summary['exactly_one_h1_rate']:.1%} "
        f"repetition_rate={summary['repetition_rate']:.1%}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
