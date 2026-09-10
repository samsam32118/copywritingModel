#!/usr/bin/env python3
"""Join judge/scores_*.jsonl against judge/key.jsonl (written by
make_judge_pack.py) and report per-system results.

key.jsonl is item_id-keyed: with --both-orders (make_judge_pack.py's
default), each sampled id contributes two items -- order "a" (random
position assignment) and order "b" (positions swapped) -- so results are
reported at two granularities:

  - "raw per-order": every scored item counts on its own (up to 2
    samples/id when both orders were judged).
  - "consistent" (the primary metric): per id, over ids judged in BOTH
    orders, a system wins the id only if it is preferred in both orders;
    any disagreement between orders (including a double tie) counts as a
    tie. This factors position bias out of the headline win rate.

Also reports per-criterion mean scores (over all items), position bias (how
often position 1 is preferred), and judge-agreement (how often both orders
of an id agree on the same system, tie-vs-tie included). Writes
judge/summary.md and judge/summary.json.

Example:
    python scripts/score_judge.py --judge-dir judge/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import eval_common as ec

# Rubric criterion keys -- kept identical (name and order) to
# make_judge_pack.py's CRITERIA, since that's the schema judges actually
# write to scores_*.jsonl.
CRITERIA = ["clarity", "specificity", "persuasiveness", "structure", "cta"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a judge pack: join scores_*.jsonl with key.jsonl and report per-system results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--judge-dir", default="judge/", help="Directory containing key.jsonl and scores_*.jsonl.")
    p.add_argument("--bootstrap-n", type=int, default=10000, help="Number of bootstrap resamples for the win-rate CI.")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed for the bootstrap.")
    return p.parse_args()


def mean(lst):
    return sum(lst) / len(lst) if lst else None


def bootstrap_ci(outcomes, n_boot, seed):
    """95% bootstrap CI for the mean of a 0/1 outcome list. (None, None) if empty."""
    if not outcomes:
        return None, None
    import numpy as np

    arr = np.asarray(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_or_na(x, spec="{:.3f}"):
    return "n/a" if x is None else spec.format(x)


def fmt_pct_or_na(x):
    return "n/a" if x is None else f"{x:.1%}"


def find_finetuned_system(systems):
    """Best-effort pick of "the fine-tuned system" among the judged system
    names, for the dedicated headline in the report. run_evals.sh always
    names it exactly "finetuned"; fall back to a substring match, else None
    -- the full per-system table below the headline still reports every
    system regardless.
    """
    for s in systems:
        if s.lower() == "finetuned":
            return s
    for s in systems:
        if "finetun" in s.lower():
            return s
    return None


def main():
    args = parse_args()

    key_path = os.path.join(args.judge_dir, "key.jsonl")
    if not os.path.exists(key_path):
        print(f"[score_judge] ERROR: {key_path} not found (run make_judge_pack.py first)", file=sys.stderr)
        sys.exit(1)
    key_rows = ec.read_jsonl(key_path)
    key = {r["item_id"]: r for r in key_rows}

    scores_files = sorted(glob.glob(os.path.join(args.judge_dir, "scores_*.jsonl")))
    if not scores_files:
        print(f"[score_judge] WARNING: no scores_*.jsonl files found under {args.judge_dir}", file=sys.stderr)

    systems = sorted({r["position_1_system"] for r in key_rows} | {r["position_2_system"] for r in key_rows})
    criterion_scores = {s: {c: [] for c in CRITERIA} for s in systems}
    raw_wins = {s: 0 for s in systems}
    raw_ties = {s: 0 for s in systems}
    raw_losses = {s: 0 for s in systems}

    n_judgments = 0  # scored items successfully joined to the key (raw, per-order)
    n_ties = 0
    n_position_1_wins = 0
    n_decisive = 0
    n_skipped_unknown_item_id = 0
    n_skipped_duplicate_item_id = 0
    seen_item_ids = set()

    # by_id[id][order] = winning system name, or "tie"; only set for items
    # with a recognized preference (1/2/tie), at most one entry per order.
    by_id = defaultdict(dict)

    for sf in scores_files:
        for row in ec.read_jsonl(sf):
            item_id = row.get("item_id")
            if item_id not in key:
                n_skipped_unknown_item_id += 1
                continue
            if item_id in seen_item_ids:
                n_skipped_duplicate_item_id += 1
                continue
            seen_item_ids.add(item_id)

            k = key[item_id]
            pid, order = k["id"], k["order"]
            pos1_name, pos2_name = k["position_1_system"], k["position_2_system"]
            n_judgments += 1

            s1, s2 = row.get("scores_1") or {}, row.get("scores_2") or {}
            for c in CRITERIA:
                if c in s1:
                    criterion_scores[pos1_name][c].append(s1[c])
                if c in s2:
                    criterion_scores[pos2_name][c].append(s2[c])

            pref = str(row.get("preference", "")).strip().lower()
            if pref == "1":
                n_position_1_wins += 1
            if pref in ("1", "2"):
                n_decisive += 1
                winner = pos1_name if pref == "1" else pos2_name
                loser = pos2_name if pref == "1" else pos1_name
                raw_wins[winner] += 1
                raw_losses[loser] += 1
                by_id[pid][order] = winner
            elif pref == "tie":
                n_ties += 1
                raw_ties[pos1_name] += 1
                raw_ties[pos2_name] += 1
                by_id[pid][order] = "tie"
            else:
                print(f"[score_judge] WARNING: item {item_id!r} has unrecognized preference {row.get('preference')!r}, skipping preference tally", file=sys.stderr)

    # --- per-system raw (per-order, not deduped by id) results ---
    per_system_raw = {}
    for s in systems:
        criterion_means = {c: mean(criterion_scores[s][c]) for c in CRITERIA}
        denom = raw_wins[s] + raw_losses[s]
        per_system_raw[s] = {
            "mean_criteria": criterion_means,
            "wins": raw_wins[s],
            "ties": raw_ties[s],
            "losses": raw_losses[s],
            "win_rate_excl_ties": (raw_wins[s] / denom) if denom else None,
        }

    # --- consistent (primary) results: per id, over ids judged in both orders ---
    ids_both_orders = [pid for pid, orders in by_id.items() if len(orders) >= 2]
    ids_single_order = [pid for pid, orders in by_id.items() if len(orders) == 1]

    consistent_wins = {s: 0 for s in systems}
    consistent_ties = {s: 0 for s in systems}
    consistent_losses = {s: 0 for s in systems}
    decisive_outcome = {s: [] for s in systems}  # 1 if s won that id, 0 if it lost that id; ties excluded

    n_agree = 0
    for pid in ids_both_orders:
        orders = by_id[pid]
        outcome_a, outcome_b = orders.get("a"), orders.get("b")
        agree = outcome_a == outcome_b
        if agree:
            n_agree += 1
        if agree and outcome_a != "tie":
            winner = outcome_a
            consistent_wins[winner] += 1
            decisive_outcome[winner].append(1)
            for loser in systems:
                if loser == winner:
                    continue
                consistent_losses[loser] += 1
                decisive_outcome[loser].append(0)
        else:
            # Disagreement between the two orders (or a double tie) both
            # count as a tie for the consistent metric: inconsistent = tie.
            for s in systems:
                consistent_ties[s] += 1

    per_system_consistent = {}
    for s in systems:
        denom = consistent_wins[s] + consistent_losses[s]
        ci_lo, ci_hi = bootstrap_ci(decisive_outcome[s], args.bootstrap_n, args.seed)
        per_system_consistent[s] = {
            "wins": consistent_wins[s],
            "ties": consistent_ties[s],
            "losses": consistent_losses[s],
            "win_rate_excl_ties": (consistent_wins[s] / denom) if denom else None,
            "win_rate_excl_ties_ci95": [ci_lo, ci_hi],
        }

    judge_agreement_rate = (n_agree / len(ids_both_orders)) if ids_both_orders else None

    position_1_win_rate_overall = (n_position_1_wins / n_judgments) if n_judgments else None
    position_1_win_rate_decisive = (n_position_1_wins / n_decisive) if n_decisive else None

    finetuned = find_finetuned_system(systems)

    summary = {
        "judge_dir": args.judge_dir,
        "n_scores_files": len(scores_files),
        "n_items_judged": n_judgments,
        "n_decisive_items": n_decisive,
        "n_tie_items": n_ties,
        "n_skipped_unknown_item_id": n_skipped_unknown_item_id,
        "n_skipped_duplicate_item_id": n_skipped_duplicate_item_id,
        "n_ids_both_orders": len(ids_both_orders),
        "n_ids_single_order": len(ids_single_order),
        "systems": systems,
        "finetuned_system": finetuned,
        "per_system_raw": per_system_raw,
        "per_system_consistent": per_system_consistent,
        "position_bias": {
            "position_1_win_rate_overall": position_1_win_rate_overall,
            "position_1_win_rate_excl_ties": position_1_win_rate_decisive,
        },
        "judge_agreement": {
            "n_ids_both_orders": len(ids_both_orders),
            "n_agree": n_agree,
            "rate": judge_agreement_rate,
        },
        "bootstrap_n": args.bootstrap_n,
        "bootstrap_seed": args.seed,
    }

    summary_json_path = os.path.join(args.judge_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- markdown report ---
    lines = ["# Judge Pack Results\n"]
    lines.append(f"Judge dir: `{args.judge_dir}`  ")
    lines.append(f"Items judged: {n_judgments} ({n_decisive} decisive, {n_ties} ties) from {len(scores_files)} scores file(s)  ")
    lines.append(f"Ids judged in both orders: {len(ids_both_orders)}  |  single order only: {len(ids_single_order)}")
    if n_skipped_unknown_item_id:
        lines.append(f"  \nSkipped {n_skipped_unknown_item_id} score row(s) with an item_id not found in key.jsonl.")
    if n_skipped_duplicate_item_id:
        lines.append(f"  \nSkipped {n_skipped_duplicate_item_id} duplicate score row(s) for an already-scored item_id.")
    lines.append("")

    lines.append("## Primary metric: consistent win rate (over ids judged in both orders)\n")
    lines.append("A system wins an id only if it's preferred in BOTH position orders; any")
    lines.append("disagreement between orders (or a double tie) counts as a tie for this metric.\n")
    if finetuned:
        lines.append(f"**Fine-tuned system: `{finetuned}`**\n")
    lines.append("| system | consistent wins | ties | losses | win rate (excl. ties) | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for s in systems:
        ps = per_system_consistent[s]
        lo, hi = ps["win_rate_excl_ties_ci95"]
        ci_str = "n/a" if lo is None else f"[{lo:.1%}, {hi:.1%}]"
        label = f"**{s}**" if s == finetuned else s
        lines.append(
            f"| {label} | {ps['wins']} | {ps['ties']} | {ps['losses']} "
            f"| {fmt_pct_or_na(ps['win_rate_excl_ties'])} | {ci_str} |"
        )

    lines.append("\n## Raw per-order results (not deduped by id)\n")
    lines.append("| system | wins | ties | losses | win rate (excl. ties) |")
    lines.append("|---|---|---|---|---|")
    for s in systems:
        ps = per_system_raw[s]
        lines.append(f"| {s} | {ps['wins']} | {ps['ties']} | {ps['losses']} | {fmt_pct_or_na(ps['win_rate_excl_ties'])} |")

    lines.append("\n## Mean rubric scores per system (1-5, over all items)\n")
    lines.append("| system | " + " | ".join(CRITERIA) + " |")
    lines.append("|---|" + "---|" * len(CRITERIA))
    for s in systems:
        mc = per_system_raw[s]["mean_criteria"]
        lines.append(f"| {s} | " + " | ".join(fmt_or_na(mc[c], "{:.2f}") for c in CRITERIA) + " |")

    lines.append("\n## Position bias\n")
    lines.append(f"- Position 1 preferred: {fmt_pct_or_na(position_1_win_rate_overall)} of all judged items")
    lines.append(f"- Position 1 preferred: {fmt_pct_or_na(position_1_win_rate_decisive)} of decisive (non-tie) items (50% = no bias)")

    lines.append("\n## Judge agreement\n")
    lines.append(
        f"- Same system-level preference in both orders: {fmt_pct_or_na(judge_agreement_rate)} "
        f"of {len(ids_both_orders)} id(s) judged in both orders (a tie in both orders counts as agreement)"
    )

    lines.append(f"\n_Bootstrap: {args.bootstrap_n} resamples, seed={args.seed}, resampled over ids._\n")

    summary_md_path = os.path.join(args.judge_dir, "summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"[score_judge] n_items={n_judgments} n_ids_both_orders={len(ids_both_orders)} "
        f"judge_agreement={fmt_pct_or_na(judge_agreement_rate)} -> {summary_md_path}",
        file=sys.stderr,
    )
    for s in systems:
        ps = per_system_consistent[s]
        print(
            f"[score_judge]   {s}: consistent_win_rate_excl_ties={fmt_pct_or_na(ps['win_rate_excl_ties'])} "
            f"(wins={ps['wins']} ties={ps['ties']} losses={ps['losses']})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
