#!/usr/bin/env python3
"""Join judge/scores_*.jsonl against judge/key.jsonl (written by
make_judge_pack.py) and report per-system results: mean score per rubric
criterion, win/tie/loss counts, win rate excluding ties, position bias, and
a 95% bootstrap CI for each system's win rate. Writes judge/summary.md.

Example:
    python scripts/score_judge.py --judge-dir judge/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import eval_common as ec

CRITERIA = ["clarity", "specificity", "persuasiveness", "structure", "cta_quality"]


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
    """95% bootstrap CI for the mean of a 0/1 outcome list. None if empty."""
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


def main():
    args = parse_args()

    key_path = os.path.join(args.judge_dir, "key.jsonl")
    if not os.path.exists(key_path):
        print(f"[score_judge] ERROR: {key_path} not found (run make_judge_pack.py first)", file=sys.stderr)
        sys.exit(1)
    key = {r["id"]: r for r in ec.read_jsonl(key_path)}

    scores_files = sorted(glob.glob(os.path.join(args.judge_dir, "scores_*.jsonl")))
    if not scores_files:
        print(f"[score_judge] WARNING: no scores_*.jsonl files found under {args.judge_dir}", file=sys.stderr)

    systems = sorted({r["position_1"] for r in key.values()} | {r["position_2"] for r in key.values()})
    criterion_scores = {sys_name: {c: [] for c in CRITERIA} for sys_name in systems}
    wins = {sys_name: 0 for sys_name in systems}
    ties = {sys_name: 0 for sys_name in systems}
    losses = {sys_name: 0 for sys_name in systems}
    decisive_outcome = {sys_name: [] for sys_name in systems}  # 1 if this system won that decisive pair, else 0

    n_judgments = 0
    n_ties = 0
    n_position_1_wins = 0
    n_decisive = 0
    n_skipped_unknown_id = 0

    for sf in scores_files:
        for row in ec.read_jsonl(sf):
            pid = row.get("id")
            if pid not in key:
                n_skipped_unknown_id += 1
                continue
            k = key[pid]
            pos1_name, pos2_name = k["position_1"], k["position_2"]
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
                wins[winner] += 1
                losses[loser] += 1
                decisive_outcome[winner].append(1)
                decisive_outcome[loser].append(0)
            elif pref == "tie":
                n_ties += 1
                ties[pos1_name] += 1
                ties[pos2_name] += 1
            else:
                print(f"[score_judge] WARNING: id {pid!r} has unrecognized preference {row.get('preference')!r}, skipping preference tally", file=sys.stderr)

    per_system = {}
    for sys_name in systems:
        criterion_means = {c: mean(criterion_scores[sys_name][c]) for c in CRITERIA}
        win_rate_excl_ties = wins[sys_name] / (wins[sys_name] + losses[sys_name]) if (wins[sys_name] + losses[sys_name]) > 0 else None
        ci_lo, ci_hi = bootstrap_ci(decisive_outcome[sys_name], args.bootstrap_n, args.seed)
        per_system[sys_name] = {
            "mean_criteria": criterion_means,
            "wins": wins[sys_name],
            "ties": ties[sys_name],
            "losses": losses[sys_name],
            "win_rate_excl_ties": win_rate_excl_ties,
            "win_rate_excl_ties_ci95": [ci_lo, ci_hi],
        }

    position_1_win_rate_overall = (n_position_1_wins / n_judgments) if n_judgments else None
    position_1_win_rate_decisive = (n_position_1_wins / n_decisive) if n_decisive else None

    summary = {
        "judge_dir": args.judge_dir,
        "n_scores_files": len(scores_files),
        "n_judgments": n_judgments,
        "n_decisive": n_decisive,
        "n_ties": n_ties,
        "n_skipped_unknown_id": n_skipped_unknown_id,
        "systems": systems,
        "per_system": per_system,
        "position_bias": {
            "position_1_win_rate_overall": position_1_win_rate_overall,
            "position_1_win_rate_excl_ties": position_1_win_rate_decisive,
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
    lines.append(f"Judgments: {n_judgments} ({n_decisive} decisive, {n_ties} ties) from {len(scores_files)} scores file(s)")
    if n_skipped_unknown_id:
        lines.append(f"  \nSkipped {n_skipped_unknown_id} score row(s) with an id not found in key.jsonl.")
    lines.append("")

    lines.append("## Per-system results\n")
    lines.append("| system | wins | ties | losses | win rate (excl. ties) | 95% CI |")
    lines.append("|---|---|---|---|---|---|")
    for sys_name in systems:
        ps = per_system[sys_name]
        lo, hi = ps["win_rate_excl_ties_ci95"]
        ci_str = "n/a" if lo is None else f"[{lo:.1%}, {hi:.1%}]"
        lines.append(
            f"| {sys_name} | {ps['wins']} | {ps['ties']} | {ps['losses']} "
            f"| {fmt_pct_or_na(ps['win_rate_excl_ties'])} | {ci_str} |"
        )

    lines.append("\n## Mean rubric scores per system (1-5)\n")
    lines.append("| system | " + " | ".join(CRITERIA) + " |")
    lines.append("|---|" + "---|" * len(CRITERIA))
    for sys_name in systems:
        mc = per_system[sys_name]["mean_criteria"]
        lines.append(f"| {sys_name} | " + " | ".join(fmt_or_na(mc[c], "{:.2f}") for c in CRITERIA) + " |")

    lines.append("\n## Position bias\n")
    lines.append(f"- Position 1 preferred: {fmt_pct_or_na(position_1_win_rate_overall)} of all judgments")
    lines.append(f"- Position 1 preferred: {fmt_pct_or_na(position_1_win_rate_decisive)} of decisive (non-tie) judgments (50% = no bias)")

    lines.append(f"\n_Bootstrap: {args.bootstrap_n} resamples, seed={args.seed}._\n")

    summary_md_path = os.path.join(args.judge_dir, "summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[score_judge] n_judgments={n_judgments} n_decisive={n_decisive} n_ties={n_ties} -> {summary_md_path}", file=sys.stderr)
    for sys_name in systems:
        ps = per_system[sys_name]
        print(
            f"[score_judge]   {sys_name}: win_rate_excl_ties={fmt_pct_or_na(ps['win_rate_excl_ties'])} "
            f"(wins={ps['wins']} ties={ps['ties']} losses={ps['losses']})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
