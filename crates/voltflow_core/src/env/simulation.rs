//! Core Gym-style step/reset state machine (TECHNICAL.md sections 4, 5).
//!
//! All buffers (prices, temps, obs scratch) are pre-allocated at construction
//! time; `step()` performs zero heap allocation in steady state (agent rule 2).

use crate::battery::cell::{clamp_soc, delta_soc, effective_power_kw, CellParams};
use crate::battery::degradation::{total_delta_soh, DegradationParams};
use crate::battery::thermal::{heat_generated_w, step_temperature, ThermalParams};
use crate::data::loader::MarketData;
use crate::env::stochastic::OuNoise;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct FinancialParams {
    pub asset_replacement_cost: f32, // $150,000
    pub t_crit: f32,                 // 318.15 K
    pub kappa: f32,                  // 50.0
    pub reward_norm_scale: f32,      // 100.0
}

impl Default for FinancialParams {
    fn default() -> Self {
        Self {
            asset_replacement_cost: 150_000.0,
            t_crit: 318.15,
            kappa: 50.0,
            reward_norm_scale: 100.0,
        }
    }
}

pub struct StepInfo {
    pub revenue: f32,
    pub degradation_cost: f32,
    pub thermal_penalty: f32,
    pub soc: f32,
    pub soh: f32,
    pub t_cell: f32,
    pub price: f32,
}

pub struct BessSimulation {
    // State
    pub soc: f32,
    pub soh: f32,
    pub t_cell: f32,
    pub current_step: usize,
    pub episode_start_idx: usize,
    pub max_steps: usize,

    // Params (frozen for the run)
    pub cell_params: CellParams,
    pub thermal_params: ThermalParams,
    pub degradation_params: DegradationParams,
    pub financial_params: FinancialParams,

    // Market data (loaded once, indexed per step - zero-copy)
    pub market: MarketData,

    // Domain randomization noise generators
    price_noise: OuNoise,
    temp_noise: OuNoise,
    rng: StdRng,

    // Pre-allocated scratch buffer for observation (avoids per-step alloc)
    obs_scratch: [f32; 8],

    pub t_ambient_initial: f32,

    // Price normalization bounds for the observation vector (TECHNICAL.md 5.1).
    // Derived from the loaded dataset itself at construction time rather
    // than hardcoded, so the observation space actually spans [0,1] for
    // whatever price distribution the CSV holds (see PRICE_NORM_PAD_FRAC
    // doc comment below for why a fixed -50/300 range is wrong for real
    // (non-negative, narrower-range) market data).
    price_norm_min: f32,
    price_norm_max: f32,
}

/// Fractional padding applied beyond the observed [min, max] price range
/// when deriving normalization bounds, so prices during deployment/eval
/// that exceed the training data's historical extremes still map inside
/// (or close to) [0, 1] instead of immediately saturating at the clamp.
const PRICE_NORM_PAD_FRAC: f32 = 0.15;

/// Floor on the normalization span width, in $/MWh, to avoid a
/// near-zero-width (or exactly zero, e.g. constant-price synthetic/test
/// data) denominator collapsing the price observation to a single value.
const PRICE_NORM_MIN_WIDTH: f32 = 10.0;

/// Design safety margin (Kelvin) kept between the worst-case sustained
/// steady-state cell temperature and `t_crit` when auto-sizing thermal
/// cooling capacity (see `derive_h_times_a`). Chosen so short-term OU noise
/// on ambient temp and transient overshoot before steady-state still don't
/// routinely cross `t_crit` even during continuous max-power operation on
/// the hottest day in the dataset.
const THERMAL_SAFETY_MARGIN_K: f32 = 10.0;

/// Absolute floor on the denominator used when deriving cooling capacity
/// from `(t_crit - margin - max_ambient)`. Without this, a dataset whose
/// hottest recorded ambient temperature sits within `THERMAL_SAFETY_MARGIN_K`
/// of `t_crit` would drive the required h*A toward +infinity. Flooring here
/// means the derived system is *undersized* for truly pathological climates
/// rather than producing a nonsensical/infinite cooling requirement — this
/// is a known, documented limitation rather than a silent failure.
const THERMAL_MIN_HEADROOM_K: f32 = 5.0;

impl BessSimulation {
    pub fn new(market: MarketData, max_steps: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);

        let (price_norm_min, price_norm_max) = Self::derive_price_norm_bounds(&market.prices);

        let cell_params = CellParams::default();
        let financial_params = FinancialParams::default();
        let mut thermal_params = ThermalParams::default();
        thermal_params.h_times_a = Self::derive_h_times_a(
            &cell_params,
            &thermal_params,
            &market.ambient_temps_k,
            financial_params.t_crit,
        );

        Self {
            soc: 0.5,
            soh: 1.0,
            t_cell: 298.15,
            current_step: 0,
            episode_start_idx: 0,
            max_steps,
            cell_params,
            thermal_params,
            degradation_params: DegradationParams::default(),
            financial_params,
            market,
            price_noise: OuNoise::new(0.3, 0.0, 3.0),
            temp_noise: OuNoise::new(0.2, 0.0, 0.5),
            rng,
            obs_scratch: [0.0; 8],
            t_ambient_initial: 288.15,
            price_norm_min,
            price_norm_max,
        }
    }

    /// Auto-sizes thermal cooling capacity (`h*A`, W/K) from the cell's own
    /// power rating and the actual ambient-temperature data being trained
    /// on, rather than using the TECHNICAL.md's flat 25.0 W/K default.
    ///
    /// KNOWN SPEC DEVIATION: the TECHNICAL.md's default h*A = 25.0 W/K is sized for
    /// a much smaller system than its own P_max = 500kW implies. At just
    /// half power (250kW action), the resulting resistive heating pushes
    /// the steady-state cell temperature ~54K above ambient, and a single
    /// 15-minute step already covers ~78% of that rise — so almost any
    /// nontrivial charge/discharge blows past T_crit (318.15K/45°C) inside
    /// one step. The resulting kappa*(T-T_crit)^2 penalty then outweighs
    /// revenue by 3-4 orders of magnitude (confirmed empirically: a single
    /// half-power step produced a thermal penalty of ~24,000 against
    /// revenue of ~-3), which would train PPO into a degenerate
    /// never-act-at-any-price policy rather than an arbitrage strategy.
    /// This isn't a training-data problem, it's a reward-shape problem, so
    /// it's fixed at the parameter level rather than patched over with
    /// reward reweighting.
    ///
    /// Fix: size h*A so that *continuous, sustained* operation at max power
    /// (the worst case — the discharge branch, where inverter loss means
    /// pack-side P_eff exceeds P_max; see `cell::effective_power_kw`) on the
    /// *hottest ambient temperature actually present in the loaded dataset*
    /// settles at steady-state `THERMAL_SAFETY_MARGIN_K` below `t_crit`,
    /// with `THERMAL_MIN_HEADROOM_K` as a floor against pathological (very
    /// hot climate) inputs. This is deliberately data- and rating-driven
    /// rather than a second hardcoded constant: swap in a different power
    /// rating, cell chemistry, or climate later and the cooling capacity
    /// re-derives itself correctly instead of needing to be hand-retuned —
    /// the same principle as `derive_price_norm_bounds` above.
    fn derive_h_times_a(
        cell: &CellParams,
        thermal: &ThermalParams,
        ambient_temps_k: &[f32],
        t_crit: f32,
    ) -> f32 {
        let eta_safe = cell.inverter_eta.max(1e-6);
        // Worst-case sustained pack-side power: the discharge branch divides
        // by eta (loss), so |P_eff| > P_max there, unlike the charge branch.
        let p_eff_max_kw = cell.max_power_kw / eta_safe;
        let q_gen_max_w = heat_generated_w(p_eff_max_kw, thermal.v_nominal, thermal.r_internal);

        let max_ambient_k = ambient_temps_k
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        // Fall back to a standard-conditions ambient (25C) if no data was
        // loaded (e.g. constructed with an empty MarketData in a test).
        let max_ambient_k = if max_ambient_k.is_finite() {
            max_ambient_k
        } else {
            298.15
        };

        let headroom_k =
            (t_crit - THERMAL_SAFETY_MARGIN_K - max_ambient_k).max(THERMAL_MIN_HEADROOM_K);

        q_gen_max_w / headroom_k
    }

    /// Computes [min, max] observation-normalization bounds from the actual
    /// price series, padded by PRICE_NORM_PAD_FRAC on each side and floored
    /// to PRICE_NORM_MIN_WIDTH total span. Falls back to the TECHNICAL.md's original
    /// -50.0/300.0 bounds if `prices` is empty (shouldn't happen in practice
    /// — the loader errors on an empty CSV — but keeps this fn total).
    fn derive_price_norm_bounds(prices: &[f32]) -> (f32, f32) {
        if prices.is_empty() {
            return (-50.0, 300.0);
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &p in prices {
            if p < min {
                min = p;
            }
            if p > max {
                max = p;
            }
        }
        let range = max - min;
        let pad = (range * PRICE_NORM_PAD_FRAC).max(0.0);
        let mut lo = min - pad;
        let mut hi = max + pad;
        if hi - lo < PRICE_NORM_MIN_WIDTH {
            let mid = (hi + lo) / 2.0;
            lo = mid - PRICE_NORM_MIN_WIDTH / 2.0;
            hi = mid + PRICE_NORM_MIN_WIDTH / 2.0;
        }
        (lo, hi)
    }

    /// Resets episode state. If `randomize` is true, picks a random start
    /// index in the market data (bounded so a full episode fits) and resets
    /// the OU noise processes; otherwise starts at index 0 deterministically.
    pub fn reset(&mut self, randomize: bool) -> [f32; 8] {
        self.soc = 0.5;
        self.soh = 1.0;
        self.t_cell = 298.15;
        self.current_step = 0;
        self.price_noise.reset();
        self.temp_noise.reset();

        let data_len = self.market.prices.len();
        let max_start = data_len.saturating_sub(self.max_steps + 2).max(1);

        self.episode_start_idx = if randomize && max_start > 1 {
            self.rng.gen_range(0..max_start)
        } else {
            0
        };

        self.compute_observation()
    }

    /// Advances the simulation by one step given a normalized action in
    /// [-1.0, 1.0]. Returns (observation, reward, terminated, truncated,
    /// (revenue, degradation_cost, thermal_penalty)).
    pub fn step(&mut self, action: f32) -> ([f32; 8], f32, bool, bool, StepInfo) {
        let action_clamped = action.clamp(-1.0, 1.0);
        let p_action_kw = action_clamped * self.cell_params.max_power_kw;

        // --- 4.1: Coulomb counting & inverter efficiency ---
        let p_eff = effective_power_kw(p_action_kw, self.cell_params.inverter_eta);
        let dsoc = delta_soc(
            p_eff,
            self.cell_params.dt_hours,
            self.cell_params.nominal_energy_kwh,
            self.soh,
        );
        let soc_before = self.soc;
        self.soc = clamp_soc(
            self.soc + dsoc,
            self.cell_params.soc_min,
            self.cell_params.soc_max,
        );
        // Actual delivered/absorbed power may be less than requested if SoC
        // clamps the action (hit soc_min/soc_max boundary).
        let actual_dsoc = self.soc - soc_before;

        // --- 4.2: Thermal dynamics ---
        let idx = self.current_market_idx();
        let t_ambient_base = self.market.ambient_temps_k[idx];
        let t_ambient = t_ambient_base + self.temp_noise.step(self.cell_params.dt_hours, &mut self.rng);

        let q_gen = heat_generated_w(p_eff, self.thermal_params.v_nominal, self.thermal_params.r_internal);
        self.t_cell = step_temperature(
            self.t_cell,
            t_ambient,
            q_gen,
            &self.thermal_params,
            self.cell_params.dt_hours,
        )
        .clamp(233.15, 373.15); // hard physical clamp: -40C to 100C

        // --- 4.3: Degradation ---
        let delta_soh_total = total_delta_soh(
            p_eff,
            self.cell_params.max_power_kw,
            self.t_cell,
            self.soc,
            &self.degradation_params,
            self.cell_params.dt_hours,
        );
        self.soh = (self.soh - delta_soh_total).clamp(0.0, 1.0);

        // --- 4.4: Financial ledger & reward ---
        let price_base = self.market.prices[idx];
        let price = price_base + self.price_noise.step(self.cell_params.dt_hours, &mut self.rng);

        // Reconstruct actual charged/discharged power from the SoC delta
        // actually applied (post-clamp), converting back through inverter
        // efficiency so revenue reflects grid-side energy, not internal P_eff.
        let energy_delta_kwh = actual_dsoc * self.cell_params.nominal_energy_kwh * self.soh.max(1e-6);
        let (p_charged_kwh, p_discharged_kwh) = if energy_delta_kwh > 0.0 {
            (energy_delta_kwh, 0.0)
        } else {
            (0.0, -energy_delta_kwh)
        };
        // price is $/MWh; convert kWh -> MWh for revenue calc
        let revenue = (p_discharged_kwh * price / 1000.0) - (p_charged_kwh * price / 1000.0);

        let degradation_cost = delta_soh_total * self.financial_params.asset_replacement_cost;

        let thermal_penalty = if self.t_cell > self.financial_params.t_crit {
            self.financial_params.kappa * (self.t_cell - self.financial_params.t_crit).powi(2)
        } else {
            0.0
        };

        let reward = (revenue - degradation_cost - thermal_penalty)
            / self.financial_params.reward_norm_scale.max(1e-6);

        self.current_step += 1;
        let terminated = self.soh <= 0.7; // effective end-of-life
        let truncated = self.current_step >= self.max_steps;

        let obs = self.compute_observation();

        let info = StepInfo {
            revenue,
            degradation_cost,
            thermal_penalty,
            soc: self.soc,
            soh: self.soh,
            t_cell: self.t_cell,
            price,
        };

        (obs, reward, terminated, truncated, info)
    }

    #[inline]
    fn current_market_idx(&self) -> usize {
        let idx = self.episode_start_idx + self.current_step;
        idx.min(self.market.prices.len() - 1)
    }

    #[inline]
    fn next_market_idx(&self) -> usize {
        let idx = self.episode_start_idx + self.current_step + 1;
        idx.min(self.market.prices.len() - 1)
    }

    /// Builds the 8-element normalized observation vector per TECHNICAL.md section 5.1.
    fn compute_observation(&mut self) -> [f32; 8] {
        let idx = self.current_market_idx();
        let next_idx = self.next_market_idx();

        let price_t = self.market.prices[idx];
        let price_t1 = self.market.prices[next_idx];
        let t_ambient = self.market.ambient_temps_k[idx];

        // Hour-of-day derived from step index assuming 15-min steps (96/day).
        let steps_per_day = 96.0_f32;
        let hour = (self.current_step as f32 % steps_per_day) / steps_per_day * 24.0;

        self.obs_scratch[0] = self.soc;
        self.obs_scratch[1] = ((self.soh - 0.7) / 0.3).clamp(-1.0, 1.0);
        self.obs_scratch[2] = ((self.t_cell - 273.15) / 60.0).clamp(-1.0, 1.0);
        self.obs_scratch[3] = ((t_ambient - 263.15) / 55.0).clamp(-1.0, 1.0);
        let price_span = (self.price_norm_max - self.price_norm_min).max(1e-6);
        self.obs_scratch[4] = ((price_t - self.price_norm_min) / price_span).clamp(0.0, 1.0);
        self.obs_scratch[5] = ((price_t1 - self.price_norm_min) / price_span).clamp(0.0, 1.0);
        self.obs_scratch[6] = (2.0 * PI * hour / 24.0).sin();
        self.obs_scratch[7] = (2.0 * PI * hour / 24.0).cos();

        self.obs_scratch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_market(n: usize) -> MarketData {
        MarketData {
            prices: (0..n).map(|i| 50.0 + (i % 24) as f32).collect(),
            ambient_temps_k: vec![288.15; n],
            solar_irradiance: vec![0.0; n],
        }
    }

    #[test]
    fn reset_returns_valid_observation() {
        let mut sim = BessSimulation::new(dummy_market(500), 96, 42);
        let obs = sim.reset(false);
        assert_eq!(obs.len(), 8);
        assert!(obs[0] >= 0.0 && obs[0] <= 1.0); // SoC
    }

    #[test]
    fn step_charging_increases_soc() {
        let mut sim = BessSimulation::new(dummy_market(500), 96, 42);
        sim.reset(false);
        let soc_before = sim.soc;
        let (_, _, _, _, _) = sim.step(1.0); // full charge action
        assert!(sim.soc > soc_before);
    }

    #[test]
    fn step_discharging_decreases_soc() {
        let mut sim = BessSimulation::new(dummy_market(500), 96, 42);
        sim.reset(false);
        let soc_before = sim.soc;
        let (_, _, _, _, _) = sim.step(-1.0); // full discharge action
        assert!(sim.soc < soc_before);
    }

    #[test]
    fn soh_never_exceeds_bounds() {
        let mut sim = BessSimulation::new(dummy_market(2000), 500, 1);
        sim.reset(false);
        for _ in 0..500 {
            let (_, _, term, trunc, _) = sim.step(0.8);
            assert!(sim.soh >= 0.0 && sim.soh <= 1.0);
            if term || trunc {
                break;
            }
        }
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut sim = BessSimulation::new(dummy_market(500), 10, 1);
        sim.reset(false);
        let mut truncated = false;
        for _ in 0..10 {
            let (_, _, _, trunc, _) = sim.step(0.0);
            truncated = trunc;
        }
        assert!(truncated);
    }

    // --- Dynamic price normalization (replaces old hardcoded -50/300) ---

    #[test]
    fn price_norm_bounds_derived_from_data_not_hardcoded() {
        // Real-world-shaped price series: always positive, narrow range
        // (like the merged Spain dataset: ~9 to ~117 EUR/MWh), nothing
        // close to the old TECHNICAL.md's -50/300 assumption.
        let prices: Vec<f32> = (0..1000).map(|i| 9.0 + (i % 108) as f32).collect();
        let (lo, hi) = BessSimulation::derive_price_norm_bounds(&prices);
        // Bounds should track the real data (with padding), not the old
        // fixed constants.
        assert!(lo > -50.0, "lower bound should not fall back to -50.0, got {lo}");
        assert!(hi < 300.0, "upper bound should not fall back to 300.0, got {hi}");
        assert!(lo < 9.0 && hi > 116.0, "padding should extend past observed min/max");
    }

    #[test]
    fn price_norm_bounds_have_min_width_for_constant_price_series() {
        // Degenerate case: every price identical (range = 0). Bounds must
        // not collapse to zero width and divide-by-near-zero in the
        // observation calc.
        let prices = vec![42.0_f32; 100];
        let (lo, hi) = BessSimulation::derive_price_norm_bounds(&prices);
        assert!(hi - lo >= PRICE_NORM_MIN_WIDTH - 1e-3);
        assert!(lo < 42.0 && hi > 42.0);
    }

    #[test]
    fn price_norm_bounds_fall_back_on_empty_prices() {
        let (lo, hi) = BessSimulation::derive_price_norm_bounds(&[]);
        assert_eq!((lo, hi), (-50.0, 300.0));
    }

    #[test]
    fn observation_price_fields_stay_in_unit_range_for_real_shaped_data() {
        // Regression test for the normalization bug: with real (non-negative,
        // ~9-117 EUR/MWh) price data, obs[4]/obs[5] must actually span a
        // meaningful chunk of [0, 1], not sit compressed near ~0.2 as they
        // would under the old fixed -50/300 mapping.
        let n = 2000;
        let market = MarketData {
            prices: (0..n).map(|i| 9.33 + (i % 108) as f32).collect(),
            ambient_temps_k: vec![288.15; n],
            solar_irradiance: vec![0.0; n],
        };
        let mut sim = BessSimulation::new(market, 500, 7);
        let obs = sim.reset(false);
        assert!(obs[4] >= 0.0 && obs[4] <= 1.0);
        assert!(obs[5] >= 0.0 && obs[5] <= 1.0);
        // Sweep steps and confirm the price observation actually uses a wide
        // span of [0,1] (i.e. isn't collapsed into a narrow sliver).
        let mut min_seen = f32::INFINITY;
        let mut max_seen = f32::NEG_INFINITY;
        for _ in 0..300 {
            let (obs, _, _, trunc, _) = sim.step(0.0);
            min_seen = min_seen.min(obs[4]);
            max_seen = max_seen.max(obs[4]);
            if trunc {
                break;
            }
        }
        assert!(
            max_seen - min_seen > 0.5,
            "price observation span too narrow: {min_seen}..{max_seen}"
        );
    }

    // --- Auto-sized thermal cooling capacity (replaces flat 25.0 W/K default) ---

    #[test]
    fn h_times_a_scales_up_from_technical_default_for_rated_power() {
        // The TECHNICAL.md's flat 25.0 W/K default is undersized for a 500kW-rated
        // cell (see derive_h_times_a doc comment). The derived value for a
        // reasonable (non-pathological) ambient must be well above it.
        let cell = CellParams::default();
        let thermal = ThermalParams::default();
        let ambient = vec![288.15_f32; 100]; // 15C, comfortable margin below t_crit
        let h = BessSimulation::derive_h_times_a(&cell, &thermal, &ambient, 318.15);
        assert!(
            h > thermal.h_times_a * 5.0,
            "derived h*A ({h}) should be substantially above the TECHNICAL.md's \
             undersized 25.0 W/K default for a 500kW-rated cell"
        );
    }

    #[test]
    fn h_times_a_increases_with_hotter_observed_ambient() {
        // Same power rating, hotter climate -> less headroom to t_crit ->
        // more cooling capacity required. Directionally this must increase,
        // not stay fixed (which a hardcoded constant would do regardless of
        // the data it's paired with).
        let cell = CellParams::default();
        let thermal = ThermalParams::default();
        let cool_ambient = vec![273.15_f32; 50]; // 0C
        let hot_ambient = vec![308.15_f32; 50]; // 35C
        let h_cool = BessSimulation::derive_h_times_a(&cell, &thermal, &cool_ambient, 318.15);
        let h_hot = BessSimulation::derive_h_times_a(&cell, &thermal, &hot_ambient, 318.15);
        assert!(
            h_hot > h_cool,
            "hotter climate should require more cooling capacity: cool={h_cool}, hot={h_hot}"
        );
    }

    #[test]
    fn h_times_a_respects_min_headroom_floor_on_pathological_climate() {
        // Ambient right at t_crit itself is a pathological/degenerate input
        // (headroom would be negative or zero without a floor). Must not
        // divide by zero or go negative/infinite.
        let cell = CellParams::default();
        let thermal = ThermalParams::default();
        let extreme_ambient = vec![318.15_f32; 10]; // == t_crit itself
        let h = BessSimulation::derive_h_times_a(&cell, &thermal, &extreme_ambient, 318.15);
        assert!(h.is_finite() && h > 0.0);
    }

    #[test]
    fn h_times_a_falls_back_to_standard_conditions_on_empty_ambient_data() {
        let cell = CellParams::default();
        let thermal = ThermalParams::default();
        let h = BessSimulation::derive_h_times_a(&cell, &thermal, &[], 318.15);
        assert!(h.is_finite() && h > 0.0);
    }

    #[test]
    fn sustained_max_power_stays_near_but_under_t_crit_with_derived_cooling() {
        // The actual regression this whole fix targets: continuous discharge
        // at rated max power (the worst case), on a dataset with a
        // comfortable ambient margin, should settle close to
        // (t_crit - THERMAL_SAFETY_MARGIN_K) at steady state, not blow
        // straight through t_crit within a couple of steps the way the
        // TECHNICAL.md's flat 25.0 W/K default did.
        let n = 5000;
        let market = MarketData {
            prices: vec![50.0; n],
            ambient_temps_k: vec![288.15; n], // 15C
            solar_irradiance: vec![0.0; n],
        };
        let mut sim = BessSimulation::new(market, 2000, 3);
        sim.reset(false);
        let mut last_t_cell = sim.t_cell;
        for _ in 0..200 {
            let (_, _, term, trunc, info) = sim.step(-1.0); // full discharge, worst case
            last_t_cell = info.t_cell;
            if term || trunc {
                break;
            }
        }
        // Should stabilize near t_crit - margin (308.15K), comfortably
        // below t_crit (318.15K) itself, and not pinned at the hard
        // physical clamp (373.15K/100C) the way the old params could drive
        // it toward.
        assert!(
            last_t_cell < 318.15,
            "steady-state temp under sustained max power should stay under t_crit, got {last_t_cell}"
        );
        assert!(
            (last_t_cell - 308.15).abs() < 3.0,
            "expected steady-state near t_crit - {THERMAL_SAFETY_MARGIN_K}K (308.15K), got {last_t_cell}"
        );
    }
}