#!/usr/bin/env python3
"""Build a blind, position-randomized human-judging pack comparing two
systems' predictions (e.g. base vs finetuned) on their shared example ids.

Writes:
  <out-dir>/pairs.jsonl   {"id","brief","output_1","output_2"}          (blind, for judges)
  <out-dir>/key.jsonl     {"id","position_1","position_2"}              (hidden system labels)
  <out-dir>/pack_XX.md    8 pairs per file, with a fixed scoring rubric

Judges fill in <out-dir>/scores_XX.jsonl (one line per pair; see the rubric
header in each pack_XX.md for the exact schema), which score_judge.py then
joins back against key.jsonl to compute per-system results.

Example:
    python scripts/make_judge_pack.py --a base_preds.jsonl --b ft_preds.jsonl \
        --names base,finetuned --n 40 --seed 7 --out-dir judge/
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import eval_common as ec

PAIRS_PER_PACK = 8

RUBRIC_HEADER_TEMPLATE = """# Landing Page Copy — Blind Judging Pack {pack_num}

You are comparing landing-page copy from two systems. Positions are
randomized independently for every pair below -- you do NOT know which
system produced Output 1 vs Output 2, and it can differ pair to pair.

For EACH pair, read the brief, then score EACH output 1-5 on:

- **Clarity** -- is the copy easy to understand at a glance?
- **Specificity** -- is it grounded in the brief, with no invented facts?
- **Persuasiveness** -- does it make a compelling case?
- **Structure** -- correct tag hierarchy (one h1, sensible h2 sections,
  h3/h4 features, CTAs in the right place)?
- **CTA quality** -- are the button labels clear, action-oriented calls to
  action?

Then give an **Overall preference**: `1`, `2`, or `tie`.

Record your scores in `{scores_file}`, one JSON line per pair:

    {{"id": "...", "scores_1": {{"clarity": N, "specificity": N, "persuasiveness": N, "structure": N, "cta_quality": N}}, "scores_2": {{...}}, "preference": "1"|"2"|"tie"}}

---
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a blind position-randomized human-judging pack from two prediction files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--a", required=True, help="Predictions JSONL for system A.")
    p.add_argument("--b", required=True, help="Predictions JSONL for system B.")
    p.add_argument("--names", required=True, help="Comma-separated system names for A,B, e.g. base,finetuned.")
    p.add_argument("--n", type=int, default=40, help="Number of shared ids to sample into the pack.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for id sampling and position assignment.")
    p.add_argument("--out-dir", default="judge/", help="Output directory for pairs.jsonl, key.jsonl, pack_*.md.")
    return p.parse_args()


def main():
    args = parse_args()

    names = [x.strip() for x in args.names.split(",")]
    if len(names) != 2 or not all(names):
        print(f"[make_judge_pack] ERROR: --names must be exactly two comma-separated names, got {args.names!r}", file=sys.stderr)
        sys.exit(1)
    name_a, name_b = names

    recs_a = {r["id"]: r for r in ec.read_jsonl(args.a)}
    recs_b = {r["id"]: r for r in ec.read_jsonl(args.b)}
    shared_ids = [i for i in recs_a if i in recs_b]  # preserves A's file order
    if not shared_ids:
        print(f"[make_judge_pack] ERROR: no shared ids between {args.a} and {args.b}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    k = min(args.n, len(shared_ids))
    if k < args.n:
        print(f"[make_judge_pack] WARNING: only {len(shared_ids)} shared ids available, requested --n {args.n}; using all {k}", file=sys.stderr)
    chosen_set = set(rng.sample(shared_ids, k))
    selected = [i for i in shared_ids if i in chosen_set]  # keep stable original order

    os.makedirs(args.out_dir, exist_ok=True)

    pairs, key = [], []
    for pid in selected:
        a_rec, b_rec = recs_a[pid], recs_b[pid]
        brief = a_rec.get("brief", b_rec.get("brief", ""))
        if rng.random() < 0.5:
            pos1_name, pos1_text = name_a, a_rec["prediction"]
            pos2_name, pos2_text = name_b, b_rec["prediction"]
        else:
            pos1_name, pos1_text = name_b, b_rec["prediction"]
            pos2_name, pos2_text = name_a, a_rec["prediction"]
        pairs.append({"id": pid, "brief": brief, "output_1": pos1_text, "output_2": pos2_text})
        key.append({"id": pid, "position_1": pos1_name, "position_2": pos2_name})

    pairs_path = os.path.join(args.out_dir, "pairs.jsonl")
    key_path = os.path.join(args.out_dir, "key.jsonl")
    ec.write_jsonl(pairs_path, pairs)
    ec.write_jsonl(key_path, key)

    n_packs = (len(pairs) + PAIRS_PER_PACK - 1) // PAIRS_PER_PACK
    pack_paths = []
    for pack_idx in range(n_packs):
        chunk = pairs[pack_idx * PAIRS_PER_PACK : (pack_idx + 1) * PAIRS_PER_PACK]
        pack_num = f"{pack_idx + 1:02d}"
        pack_path = os.path.join(args.out_dir, f"pack_{pack_num}.md")
        scores_file = os.path.join(args.out_dir, f"scores_{pack_num}.jsonl")

        parts = [RUBRIC_HEADER_TEMPLATE.format(pack_num=pack_num, scores_file=scores_file)]
        for pair in chunk:
            parts.append(f"## Pair `{pair['id']}`\n")
            parts.append("**Brief:**\n")
            parts.append("```\n" + pair["brief"] + "\n```\n")
            parts.append("**Output 1:**\n")
            parts.append("```\n" + pair["output_1"] + "\n```\n")
            parts.append("**Output 2:**\n")
            parts.append("```\n" + pair["output_2"] + "\n```\n")
            parts.append("---\n")

        with open(pack_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        pack_paths.append(pack_path)

    print(
        f"[make_judge_pack] {len(pairs)} pairs from {len(shared_ids)} shared ids "
        f"({name_a} vs {name_b}, seed={args.seed}) -> {pairs_path}, {key_path}, {n_packs} pack file(s)",
        file=sys.stderr,
    )
    for pp in pack_paths:
        print(f"[make_judge_pack]   {pp}", file=sys.stderr)


if __name__ == "__main__":
    main()
