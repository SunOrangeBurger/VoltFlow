//! CSV ingestion into contiguous in-memory buffers for zero-copy simulation.
//!
//! Expected CSV schema (header row required, column order does not matter):
//!   timestamp        - ISO8601 or any string (unused, kept for human reference)
//!   price_eur_mwh     - spot price, $/MWh  (float)
//!   ambient_temp_c    - ambient temperature, degrees Celsius (float)
//!   solar_irradiance  - W/m^2 (float, optional; defaults to 0.0 if column absent)
//!
//! See `data/raw/energy_weather_spain.csv` for the exact column names this
//! loader expects, and README.md for where to obtain the real Kaggle dataset
//! (see progress.md dataset section for schema mapping to the Kaggle source).

use csv::ReaderBuilder;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct LoaderError(pub String);

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VoltFlow data loader error: {}", self.0)
    }
}
impl Error for LoaderError {}

pub struct MarketData {
    pub prices: Vec<f32>,         // $/MWh
    pub ambient_temps_k: Vec<f32>, // Kelvin (converted from Celsius on load)
    pub solar_irradiance: Vec<f32>,
}

/// Loads the CSV at `path` into three parallel, contiguous Vec<f32> buffers.
/// Rows with unparseable numeric fields are skipped with a warning rather
/// than panicking (no unwrap() in production paths per agent directives).
pub fn load_market_csv(path: &str) -> Result<MarketData, LoaderError> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| LoaderError(format!("failed to open '{}': {}", path, e)))?;

    let headers = reader
        .headers()
        .map_err(|e| LoaderError(format!("failed to read headers: {}", e)))?
        .clone();

    let price_idx = find_column(&headers, &["price_eur_mwh", "price", "spot_price"])?;
    let temp_idx = find_column(&headers, &["ambient_temp_c", "temp_ambient_c", "temperature"])?;
    let solar_idx = find_column(&headers, &["solar_irradiance", "ghi", "solar"]).ok();

    let mut prices = Vec::new();
    let mut ambient_temps_k = Vec::new();
    let mut solar_irradiance = Vec::new();

    for (row_num, result) in reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("VoltFlow loader: skipping malformed row {}: {}", row_num, e);
                continue;
            }
        };

        let price: f32 = match record.get(price_idx).and_then(|v| v.trim().parse().ok()) {
            Some(v) => v,
            None => {
                eprintln!("VoltFlow loader: skipping row {} (bad price)", row_num);
                continue;
            }
        };

        let temp_c: f32 = match record.get(temp_idx).and_then(|v| v.trim().parse().ok()) {
            Some(v) => v,
            None => {
                eprintln!("VoltFlow loader: skipping row {} (bad temp)", row_num);
                continue;
            }
        };

        let solar: f32 = solar_idx
            .and_then(|idx| record.get(idx))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.0);

        prices.push(price);
        ambient_temps_k.push(temp_c + 273.15);
        solar_irradiance.push(solar);
    }

    if prices.is_empty() {
        return Err(LoaderError(
            "no valid rows parsed from CSV; check column names against loader.rs schema".into(),
        ));
    }

    Ok(MarketData {
        prices,
        ambient_temps_k,
        solar_irradiance,
    })
}

fn find_column(headers: &csv::StringRecord, candidates: &[&str]) -> Result<usize, LoaderError> {
    for candidate in candidates {
        if let Some(idx) = headers.iter().position(|h| h.trim().eq_ignore_ascii_case(candidate)) {
            return Ok(idx);
        }
    }
    Err(LoaderError(format!(
        "none of the expected columns {:?} found in CSV header {:?}",
        candidates,
        headers.iter().collect::<Vec<_>>()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_valid_csv() {
        let mut file = tempfile_with_content(
            "timestamp,price_eur_mwh,ambient_temp_c,solar_irradiance\n\
             2023-01-01T00:00,45.2,10.5,0.0\n\
             2023-01-01T01:00,42.1,10.1,0.0\n",
        );
        let path = file.path().to_str().unwrap().to_string();
        file.flush().unwrap();
        let data = load_market_csv(&path).expect("should load");
        assert_eq!(data.prices.len(), 2);
        assert!((data.ambient_temps_k[0] - 283.65).abs() < 0.01);
    }

    #[test]
    fn errors_on_missing_columns() {
        let mut file = tempfile_with_content("foo,bar\n1,2\n");
        let path = file.path().to_str().unwrap().to_string();
        file.flush().unwrap();
        assert!(load_market_csv(&path).is_err());
    }

    fn tempfile_with_content(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }
}
