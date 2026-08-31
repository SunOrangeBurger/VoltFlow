//! Core Gym-style step/reset state machine (spec sections 4, 5).
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
}

impl BessSimulation {
    pub fn new(market: MarketData, max_steps: usize, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self {
            soc: 0.5,
            soh: 1.0,
            t_cell: 298.15,
            current_step: 0,
            episode_start_idx: 0,
            max_steps,
            cell_params: CellParams::default(),
            thermal_params: ThermalParams::default(),
            degradation_params: DegradationParams::default(),
            financial_params: FinancialParams::default(),
            market,
            price_noise: OuNoise::new(0.3, 0.0, 3.0),
            temp_noise: OuNoise::new(0.2, 0.0, 0.5),
            rng,
            obs_scratch: [0.0; 8],
            t_ambient_initial: 288.15,
        }
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

    /// Builds the 8-element normalized observation vector per spec section 5.1.
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
        self.obs_scratch[4] = ((price_t.clamp(-50.0, 300.0) + 50.0) / 350.0).clamp(0.0, 1.0);
        self.obs_scratch[5] = ((price_t1.clamp(-50.0, 300.0) + 50.0) / 350.0).clamp(0.0, 1.0);
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
}
