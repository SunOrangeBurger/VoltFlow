"""Generates a synthetic, schema-compatible CSV for testing the VoltFlow
pipeline end-to-end WITHOUT the real Kaggle dataset.

This is a placeholder only. Do not use for real training/benchmark claims --
swap in the real Kaggle-derived data/raw/energy_weather_spain.csv (see
download_data.py) before running Phase 4 training for real.

Produces a daily price cycle (cheap at night, expensive evening peak) plus a
seasonal + diurnal ambient temperature curve, with light gaussian noise, at
15-minute resolution -- matching the 96-steps/day convention used throughout
the spec.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate(n_days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps_per_day = 96  # 15-minute resolution
    n_steps = n_days * steps_per_day

    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=15 * i) for i in range(n_steps)]

    hours = np.array([t.hour + t.minute / 60.0 for t in timestamps])
    day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])

    # Price: base + diurnal double-peak (morning + evening) + seasonal + noise
    diurnal = 20.0 * np.sin((hours - 6.0) / 24.0 * 2 * np.pi) + 15.0 * np.sin(
        (hours - 18.0) / 12.0 * 2 * np.pi
    ).clip(min=0)
    seasonal_price = 10.0 * np.sin(day_of_year / 365.0 * 2 * np.pi)
    price = 60.0 + diurnal + seasonal_price + rng.normal(0, 5.0, n_steps)

    # Ambient temp: seasonal (winter/summer) + diurnal swing + noise, Celsius
    seasonal_temp = 10.0 * np.sin((day_of_year - 80) / 365.0 * 2 * np.pi)
    diurnal_temp = 6.0 * np.sin((hours - 9.0) / 24.0 * 2 * np.pi)
    ambient_temp_c = 15.0 + seasonal_temp + diurnal_temp + rng.normal(0, 1.0, n_steps)

    # Solar irradiance: simple daylight bell curve, zero at night
    solar = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0, None) * 800.0
    solar += rng.normal(0, 20.0, n_steps).clip(min=0)
    solar = np.where((hours < 6) | (hours > 18), 0.0, solar)

    return pd.DataFrame(
        {
            "timestamp": [t.isoformat() for t in timestamps],
            "price_eur_mwh": price,
            "ambient_temp_c": ambient_temp_c,
            "solar_irradiance": solar,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/energy_weather_spain.csv")
    args = parser.parse_args()

    df = generate(args.days, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic rows ({args.days} days) to {args.out}")


if __name__ == "__main__":
    main()
