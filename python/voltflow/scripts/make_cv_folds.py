"""Builds walk-forward (rolling-origin) CV fold CSVs from the full VoltFlow
dataset, so each fold's eval year is never seen during that fold's training.

Folds (calendar years, Europe/Madrid local time as already used by the
source CSV's `timestamp` column):

    Fold 1: train 2015           -> eval 2016
    Fold 2: train 2015-2016      -> eval 2017
    Fold 3: train 2015-2017      -> eval 2018

Writes to data/cv/foldN_{train,eval}.csv. Safe to re-run; overwrites in place.

Usage:
    python python/voltflow/scripts/make_cv_folds.py \
        --csv data/raw/energy_weather_spain.csv \
        --out-dir data/cv
"""

from __future__ import annotations

import argparse
import csv
import os


FOLDS = [
    # (fold_name, train_years, eval_year)
    ("fold1", [2015], 2016),
    ("fold2", [2015, 2016], 2017),
    ("fold3", [2015, 2016, 2017], 2018),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/energy_weather_spain.csv")
    parser.add_argument("--out-dir", type=str, default="data/cv")
    args = parser.parse_args()

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    def year_of(row):
        return int(row["timestamp"][:4])

    os.makedirs(args.out_dir, exist_ok=True)

    for fold_name, train_years, eval_year in FOLDS:
        train_rows = [r for r in rows if year_of(r) in train_years]
        eval_rows = [r for r in rows if year_of(r) == eval_year]

        train_path = os.path.join(args.out_dir, f"{fold_name}_train.csv")
        eval_path = os.path.join(args.out_dir, f"{fold_name}_eval.csv")

        for path, subset in ((train_path, train_rows), (eval_path, eval_rows)):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(subset)

        print(
            f"{fold_name}: train={train_years} ({len(train_rows)} rows) -> "
            f"{train_path}, eval={eval_year} ({len(eval_rows)} rows) -> {eval_path}"
        )

    print("\nDone. Verify row counts look sane (roughly 8760/year, "
          "8784 for leap years) before training.")


if __name__ == "__main__":
    main()