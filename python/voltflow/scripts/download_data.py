"""Fetches and validates the real market/weather dataset for VoltFlow.

DATASET DECISION (see progress.md for full rationale):
  Use ONLY the Kaggle "Energy Consumption, Generation, Prices and Weather"
  dataset (Spain, 2015-2018, hourly):
  https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

  CityLearn is NOT used: it's a multi-building demand-response benchmark
  with a different observation/action contract, not a single-BESS arbitrage
  price+weather time series. Wrong shape for this spec.

This machine's sandbox cannot reach kaggle.com directly (network not
whitelisted). Run this script on YOUR local machine after:

  1. pip install kaggle
  2. Place your Kaggle API token at ~/.kaggle/kaggle.json
     (get it from https://www.kaggle.com/settings -> API -> Create New Token)
  3. python python/voltflow/scripts/download_data.py

This script downloads the raw Kaggle CSVs, merges energy + weather data,
and writes the VoltFlow-schema CSV to data/raw/energy_weather_spain.csv
which the Rust loader (data/loader.rs) expects:
    timestamp, price_eur_mwh, ambient_temp_c, solar_irradiance

If you don't have Kaggle access, use generate_synthetic_data.py instead
(same repo, produces a schema-compatible placeholder CSV for local testing
of the pipeline only -- do NOT use synthetic data for real training claims).
"""

from __future__ import annotations

import argparse
import os
import zipfile

import pandas as pd

KAGGLE_DATASET_SLUG = "nicholasjhana/energy-consumption-generation-prices-and-weather"
OUTPUT_PATH = "data/raw/energy_weather_spain.csv"


def download_via_kaggle_cli(dest_dir: str) -> None:
    """Requires `kaggle` CLI installed and authenticated (~/.kaggle/kaggle.json)."""
    import subprocess

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_SLUG, "-p", dest_dir],
        check=True,
    )
    zip_path = os.path.join(dest_dir, "energy-consumption-generation-prices-and-weather.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)


def transform_to_voltflow_schema(raw_dir: str, output_path: str) -> None:
    """Merges the Kaggle energy_dataset.csv and weather_features.csv into
    the flat schema VoltFlow's Rust loader expects.

    Kaggle energy_dataset.csv relevant columns:
        time, price actual (EUR/MWh)
    Kaggle weather_features.csv relevant columns:
        dt_iso, city_name, temp (Kelvin, needs -273.15 for Celsius),
        (Madrid rows used as the representative single-site ambient temp)
    """
    energy_path = os.path.join(raw_dir, "energy_dataset.csv")
    weather_path = os.path.join(raw_dir, "weather_features.csv")

    if not (os.path.exists(energy_path) and os.path.exists(weather_path)):
        raise FileNotFoundError(
            f"Expected {energy_path} and {weather_path} after Kaggle download. "
            "Check the extracted zip contents match these filenames."
        )

    energy = pd.read_csv(energy_path, parse_dates=["time"])
    weather = pd.read_csv(weather_path, parse_dates=["dt_iso"])

    weather_madrid = weather[weather["city_name"] == "Madrid"].copy()
    weather_madrid["dt_iso"] = pd.to_datetime(weather_madrid["dt_iso"], utc=True).dt.tz_localize(None)
    energy["time"] = pd.to_datetime(energy["time"], utc=True).dt.tz_localize(None)

    merged = pd.merge(
        energy[["time", "price actual"]],
        weather_madrid[["dt_iso", "temp"]],
        left_on="time",
        right_on="dt_iso",
        how="inner",
    )

    out = pd.DataFrame(
        {
            "timestamp": merged["time"],
            "price_eur_mwh": merged["price actual"],
            "ambient_temp_c": merged["temp"] - 273.15,  # Kaggle temp is in Kelvin
            "solar_irradiance": 0.0,  # not present in this dataset; defaulted
        }
    )
    out = out.dropna().reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw/_kaggle_download")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    parser.add_argument("--skip-download", action="store_true", help="Use already-downloaded files")
    args = parser.parse_args()

    if not args.skip_download:
        download_via_kaggle_cli(args.raw_dir)

    transform_to_voltflow_schema(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
