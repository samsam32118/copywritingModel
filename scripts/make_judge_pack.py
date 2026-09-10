#!/usr/bin/env python3
"""Build a blind, position-randomized human-judging pack comparing two
systems' predictions (e.g. base vs finetuned) on their shared example ids.

With --both-orders (the default), each sampled id produces TWO judging
items sharing that id: order "a" (positions randomly assigned, as before)
and order "b" (positions swapped relative to "a"). This lets score_judge.py
compute a position-bias-controlled "consistent win rate" -- a system only
wins an id if it's preferred in BOTH position orders. The two orders of the
same id are always placed in different pack_XX.md files, so no single pack
shows a judge the same id twice.

Writes:
  <out-dir>/pairs.jsonl   {"item_id","id","order","brief","output_1","output_2"}          (blind, for judges)
  <out-dir>/key.jsonl     {"item_id","id","order","position_1_system","position_2_system"} (hidden system labels)
  <out-dir>/pack_XX.md    up to 8 items per file, with a fixed scoring rubric
  <out-dir>/README.md     instructions for a judge on how to fill a pack

Judges fill in <out-dir>/scores_XX.jsonl (one line per item; see the rubric
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

ITEMS_PER_PACK = 8

# Rubric criterion keys, in the order judges should score them. Shared
# (by convention -- these two files intentionally keep an identical literal
# list) with score_judge.py's CRITERIA, so the schema in the pack docs can
# never drift from what score_judge.py actually parses.
CRITERIA = ["clarity", "specificity", "persuasiveness", "structure", "cta"]
CRITERIA_EXAMPLE = ", ".join(f'"{c}": N' for c in CRITERIA)

RUBRIC_HEADER_TEMPLATE = """# Landing Page Copy — Blind Judging Pack {pack_num}

You are comparing landing-page copy from two systems. Positions are
randomized independently for every item below -- you do NOT know which
system produced Output 1 vs Output 2, and it can differ item to item. Some
ids appear in more than one pack, once with each position order, but you
will never see both copies of the same id in this pack.

For EACH item, read the brief, then score EACH output 1-5 on:

- **Clarity** -- is the copy easy to understand at a glance?
- **Specificity** -- is it grounded in the brief, with no invented facts?
- **Persuasiveness** -- does it make a compelling case?
- **Structure** -- correct tag hierarchy: exactly one h1, sensible h2/h3/h4
  sections, and CTAs in the right place?
- **CTA quality** -- are the button labels clear, action-oriented calls to
  action?

Then give an **Overall preference**: `1`, `2`, or `tie`.

Record your scores in `{scores_file}`, one JSON line per item, keyed by the
`item_id` printed above each item below (NOT the bare id), in exactly this
format:

    {{"item_id": "<id>#<order>", "scores_1": {{{criteria_example}}}, "scores_2": {{{criteria_example}}}, "preference": "1"|"2"|"tie", "note": "<=20 words"}}

`note` is a short (<=20 words) free-text justification for your preference;
every other field is required. For example, the first item below would be
`{example_item_id}`.

---
"""

README_TEMPLATE = """# Judge Instructions

Thanks for judging! This pack compares landing-page copy from two systems,
`{name_a}` and `{name_b}`, though you won't see those names anywhere below
-- everything is blind and position-randomized.

## Files

- `pack_01.md`, `pack_02.md`, ... -- the items to judge, up to {items_per_pack}
  per file, each with a brief and two candidate outputs.
- `scores_01.jsonl`, `scores_02.jsonl`, ... -- where you write your scores,
  one file per pack, one JSON line per item (`pack_07.md` -> `scores_07.jsonl`).
- `pairs.jsonl` / `key.jsonl` -- machine-readable versions of the same data;
  `key.jsonl` reveals which system produced which position, so don't open it
  before (or while) judging.

## What to do

1. Open a `pack_XX.md` file and work through it top to bottom.
2. For each item, read the brief, then read both outputs.
3. Score each output 1-5 on the five rubric criteria printed at the top of
   the pack (clarity, specificity, persuasiveness, structure, CTA quality),
   then give an overall preference of `1`, `2`, or `tie`.
4. Append one JSON line per item to the matching `scores_XX.jsonl`, using
   the exact schema shown at the top of the pack -- keyed by `item_id`, not
   the bare `id`.
5. Repeat for every pack file. Finish a pack in one sitting where you can --
   don't go back and revise earlier items after seeing later ones.

## A few things worth knowing

- Some ids appear twice across the whole set of packs, once with each
  position order, so position bias can be measured and factored out of the
  results. The two copies are always in different pack files and are not
  flagged as a pair anywhere in the pack -- judge each item independently,
  on its own merits, as if you'd never seen the brief before. Don't try to
  spot or match them up.
- Positions are independently randomized per item; which system is Output 1
  vs Output 2 can and will flip between items in the same pack.
- Do not open `key.jsonl` until you're completely done judging -- it
  reveals the hidden system identities and would un-blind you.
- If an output is malformed, incomplete, or clearly broken, judge it on
  merit as-is (score it low) rather than skipping it.

## After judging

Once every pack has a matching `scores_XX.jsonl`, run:

    python scripts/score_judge.py --judge-dir {out_dir}

to produce `summary.md` / `summary.json` with per-system results.
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
    p.add_argument(
        "--both-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit both position orders (a: random, b: swapped) per sampled id, so score_judge.py "
        "can compute a position-bias-controlled consistent win rate. Pass --no-both-orders for one "
        "randomized item per id instead (legacy behavior; no consistent-win-rate signal).",
    )
    return p.parse_args()


def build_item(pid: str, order: str, brief: str, pos1_name: str, pos1_text: str, pos2_name: str, pos2_text: str):
    """One judging item: the blind pair (for pairs.jsonl/the pack markdown)
    and its hidden key row (for key.jsonl), sharing the same item_id."""
    item_id = f"{pid}#{order}"
    pair = {"item_id": item_id, "id": pid, "order": order, "brief": brief, "output_1": pos1_text, "output_2": pos2_text}
    key = {"item_id": item_id, "id": pid, "order": order, "position_1_system": pos1_name, "position_2_system": pos2_name}
    return pair, key


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

    items_a, keys_a = [], []
    items_b, keys_b = [], []
    for pid in selected:
        a_rec, b_rec = recs_a[pid], recs_b[pid]
        brief = a_rec.get("brief", b_rec.get("brief", ""))
        if rng.random() < 0.5:
            pos1_name, pos1_text = name_a, a_rec["prediction"]
            pos2_name, pos2_text = name_b, b_rec["prediction"]
        else:
            pos1_name, pos1_text = name_b, b_rec["prediction"]
            pos2_name, pos2_text = name_a, a_rec["prediction"]

        pair_a, key_a = build_item(pid, "a", brief, pos1_name, pos1_text, pos2_name, pos2_text)
        items_a.append(pair_a)
        keys_a.append(key_a)

        if args.both_orders:
            # Order "b": same brief, positions swapped relative to order "a"
            # (not re-randomized) -- each system lands in each position
            # exactly once across the pair of orders for this id.
            pair_b, key_b = build_item(pid, "b", brief, pos2_name, pos2_text, pos1_name, pos1_text)
            items_b.append(pair_b)
            keys_b.append(key_b)

    pairs = items_a + items_b
    key = keys_a + keys_b

    pairs_path = os.path.join(args.out_dir, "pairs.jsonl")
    key_path = os.path.join(args.out_dir, "key.jsonl")
    ec.write_jsonl(pairs_path, pairs)
    ec.write_jsonl(key_path, key)

    # Chunk order-"a" and order-"b" items into packs SEPARATELY, then
    # interleave the resulting packs. A pack built from items_a can then
    # never contain items_b's copy of the same id (and vice versa) -- the
    # two orders of one id are guaranteed to land in different pack files.
    def chunk(items):
        return [items[i : i + ITEMS_PER_PACK] for i in range(0, len(items), ITEMS_PER_PACK)]

    chunks_a, chunks_b = chunk(items_a), chunk(items_b)
    interleaved = []
    for i in range(max(len(chunks_a), len(chunks_b))):
        if i < len(chunks_a):
            interleaved.append(chunks_a[i])
        if i < len(chunks_b):
            interleaved.append(chunks_b[i])

    pack_paths = []
    for pack_idx, chunk_items in enumerate(interleaved):
        pack_num = f"{pack_idx + 1:02d}"
        pack_path = os.path.join(args.out_dir, f"pack_{pack_num}.md")
        scores_file = os.path.join(args.out_dir, f"scores_{pack_num}.jsonl")

        parts = [
            RUBRIC_HEADER_TEMPLATE.format(
                pack_num=pack_num,
                scores_file=scores_file,
                criteria_example=CRITERIA_EXAMPLE,
                example_item_id=chunk_items[0]["item_id"],
            )
        ]
        for item in chunk_items:
            parts.append(f"## Item `{item['item_id']}`\n")
            parts.append("**Brief:**\n")
            parts.append("```\n" + item["brief"] + "\n```\n")
            parts.append("**Output 1:**\n")
            parts.append("```\n" + item["output_1"] + "\n```\n")
            parts.append("**Output 2:**\n")
            parts.append("```\n" + item["output_2"] + "\n```\n")
            parts.append("---\n")

        with open(pack_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        pack_paths.append(pack_path)

    readme_path = os.path.join(args.out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE.format(name_a=name_a, name_b=name_b, items_per_pack=ITEMS_PER_PACK, out_dir=args.out_dir))

    print(
        f"[make_judge_pack] {len(selected)} ids -> {len(pairs)} items (both_orders={args.both_orders}) "
        f"from {len(shared_ids)} shared ids ({name_a} vs {name_b}, seed={args.seed}) "
        f"-> {pairs_path}, {key_path}, {len(pack_paths)} pack file(s), {readme_path}",
        file=sys.stderr,
    )
    for pp in pack_paths:
        print(f"[make_judge_pack]   {pp}", file=sys.stderr)


if __name__ == "__main__":
    main()
