"""Runs the full walk-forward CV sweep: 3 folds x 5 seeds = 15 PPO training
runs, each benchmarked only on its own fold's held-out year, then aggregates
everything into a single cross-validation summary.

Prerequisite: run make_cv_folds.py first to produce data/cv/foldN_{train,eval}.csv.

This script shells out to train_ppo.py and run_benchmarks.py per run rather
than importing them directly, so a crash/OOM on one run doesn't take down
the whole sweep -- each run's failure is caught and logged, the sweep moves
on to the next.

Usage:
    python python/voltflow/scripts/run_cv_sweep.py \
        --seeds 1 2 3 4 5 \
        --timesteps 2000000 \
        --n-eval-episodes 5

This will take a long time (15x a single training run) -- expect this to
run for many hours to multiple days depending on your hardware. Consider
running under `nohup` / `tmux` / `screen`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

FOLDS = ["fold1", "fold2", "fold3"]
FOLD_EVAL_YEAR = {"fold1": 2016, "fold2": 2017, "fold3": 2018}


def run(cmd: list[str], log_path: str) -> bool:
    print(f"\n$ {' '.join(cmd)}")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"  -> {status}, log: {log_path}")
    return ok


def parse_pnl_from_report(md_path: str) -> dict | None:
    """Pulls the PPO row's mean net PnL back out of a run_benchmarks.py
    Markdown report. Returns None if the PPO row isn't present (e.g. the
    training run for this fold/seed failed)."""
    if not os.path.exists(md_path):
        return None
    with open(md_path) as f:
        for line in f:
            if line.startswith("| VoltFlow RL (PPO)"):
                cells = [c.strip() for c in line.strip("|\n").split("|")]
                # cells: [name, "mean±std", "mean±std", "mean±std", soh, "mean±std"]
                pnl_mean_str = cells[1].split("±")[0]
                return {"net_pnl_mean": float(pnl_mean_str)}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir", type=str, default="data/cv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--models-dir", type=str, default="models/cv")
    parser.add_argument("--logs-dir", type=str, default="logs/cv")
    parser.add_argument("--results-dir", type=str, default="results/cv")
    parser.add_argument("--out", type=str, default="benchmark_results_cv_summary.md")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    all_results = {}  # fold -> seed -> {"net_pnl_mean": float} or None

    for fold in FOLDS:
        train_csv = os.path.join(args.cv_dir, f"{fold}_train.csv")
        eval_csv = os.path.join(args.cv_dir, f"{fold}_eval.csv")
        if not (os.path.exists(train_csv) and os.path.exists(eval_csv)):
            print(f"Missing {train_csv} or {eval_csv} -- run make_cv_folds.py first.")
            sys.exit(1)

        all_results[fold] = {}

        for seed in args.seeds:
            tag = f"{fold}_seed{seed}"
            model_out = os.path.join(args.models_dir, f"ppo_voltflow_{tag}")
            tb_log = os.path.join(args.logs_dir, tag)
            train_log = os.path.join(args.logs_dir, f"{tag}_train.log")

            train_ok = run(
                [
                    sys.executable, "-m", "voltflow.models.train_ppo",
                    "--csv", train_csv,
                    "--timesteps", str(args.timesteps),
                    "--seed", str(seed),
                    "--out", model_out,
                    "--tensorboard-log", tb_log,
                ],
                train_log,
            )

            if not train_ok:
                all_results[fold][seed] = None
                continue

            bench_out = os.path.join(args.results_dir, f"{tag}.md")
            bench_log = os.path.join(args.logs_dir, f"{tag}_bench.log")
            bench_ok = run(
                [
                    sys.executable, "python/voltflow/scripts/run_benchmarks.py",
                    "--csv", eval_csv,
                    "--ppo-model", model_out + ".zip",
                    "--days", str(args.eval_days),
                    "--n-episodes", str(args.n_eval_episodes),
                    "--seed", "123",
                    "--out", bench_out,
                ],
                bench_log,
            )

            all_results[fold][seed] = parse_pnl_from_report(bench_out) if bench_ok else None

    # --- Aggregate ---
    lines = [
        "# VoltFlow Walk-Forward CV Summary",
        "",
        f"3 folds x {len(args.seeds)} seeds. Each fold trains only on years "
        f"before its eval year (never seen during that fold's training).",
        "",
        "| Fold | Train years | Eval year | Seeds OK | Net PnL mean±std ($) | Min | Max |",
        "|---|---|---|---|---|---|---|",
    ]

    fold_train_years = {"fold1": "2015", "fold2": "2015-2016", "fold3": "2015-2017"}

    overall_pnls = []
    for fold in FOLDS:
        pnls = [v["net_pnl_mean"] for v in all_results[fold].values() if v is not None]
        n_ok = len(pnls)
        n_total = len(args.seeds)
        if pnls:
            arr = np.array(pnls)
            mean, std, mn, mx = arr.mean(), arr.std(), arr.min(), arr.max()
            overall_pnls.extend(pnls)
            lines.append(
                f"| {fold} | {fold_train_years[fold]} | {FOLD_EVAL_YEAR[fold]} | "
                f"{n_ok}/{n_total} | {mean:.2f}±{std:.2f} | {mn:.2f} | {mx:.2f} |"
            )
        else:
            lines.append(
                f"| {fold} | {fold_train_years[fold]} | {FOLD_EVAL_YEAR[fold]} | "
                f"0/{n_total} | -- (all runs failed, check logs/cv/) | -- | -- |"
            )

    if overall_pnls:
        arr = np.array(overall_pnls)
        lines.append("")
        lines.append(
            f"**Overall across all {len(overall_pnls)} successful (fold, seed) runs: "
            f"mean net PnL ${arr.mean():.2f} ± ${arr.std():.2f}** "
            f"(min ${arr.min():.2f}, max ${arr.max():.2f})"
        )
        lines.append("")
        lines.append(
            "Compare this distribution against the heuristic baselines' PnL on the "
            "same held-out years (see per-run reports in `results/cv/*.md`) to judge "
            "genuine generalization rather than a single-seed result."
        )

    lines.append("")
    lines.append("## Raw per-run results")
    lines.append("")
    lines.append("| Fold | Seed | Net PnL ($) | Status |")
    lines.append("|---|---|---|---|")
    for fold in FOLDS:
        for seed in args.seeds:
            v = all_results[fold].get(seed)
            if v is None:
                lines.append(f"| {fold} | {seed} | -- | FAILED |")
            else:
                lines.append(f"| {fold} | {seed} | {v['net_pnl_mean']:.2f} | OK |")

    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nWritten to {args.out}")

    # Also dump raw JSON for programmatic downstream use (e.g. frontend model picker).
    json_out = args.out.replace(".md", ".json")
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results also written to {json_out}")


if __name__ == "__main__":
    main()