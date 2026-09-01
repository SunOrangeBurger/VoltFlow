//! Non-linear cycle + calendar aging degradation model (TECHNICAL.md section 4.3).

#[derive(Debug, Clone, Copy)]
pub struct DegradationParams {
    pub gamma_cycle: f32, // 1.2e-6
    pub gamma_cal: f32,   // 4.0e-7
    pub k_temp: f32,      // 0.03
}

impl Default for DegradationParams {
    fn default() -> Self {
        Self {
            gamma_cycle: 1.2e-6,
            gamma_cal: 4.0e-7,
            k_temp: 0.03,
        }
    }
}

const T_REF_KELVIN: f32 = 298.15;

/// Cycle aging component: driven by |P_eff|/P_max raised to 1.3, with an
/// Arrhenius-like temperature acceleration term.
#[inline]
pub fn delta_soh_cycle(
    p_eff_kw: f32,
    p_max_kw: f32,
    t_cell: f32,
    gamma_cycle: f32,
    dt_hours: f32,
) -> f32 {
    let p_max_safe = p_max_kw.max(1e-6);
    let power_ratio = (p_eff_kw.abs() / p_max_safe).powf(1.3);
    let temp_accel = ((t_cell - T_REF_KELVIN) / 20.0).exp();
    gamma_cycle * power_ratio * temp_accel * dt_hours
}

/// Calendar aging component: driven by temperature and SoC^0.5 (higher SoC
/// accelerates calendar fade, as is physically typical for Li-ion).
#[inline]
pub fn delta_soh_calendar(
    t_cell: f32,
    soc: f32,
    gamma_cal: f32,
    k_temp: f32,
    dt_hours: f32,
) -> f32 {
    let temp_accel = (k_temp * (t_cell - T_REF_KELVIN)).exp();
    let soc_term = soc.max(0.0).sqrt();
    gamma_cal * temp_accel * soc_term * dt_hours
}

/// Total SoH loss for this step (always >= 0; caller subtracts from SoH).
#[inline]
pub fn total_delta_soh(
    p_eff_kw: f32,
    p_max_kw: f32,
    t_cell: f32,
    soc: f32,
    params: &DegradationParams,
    dt_hours: f32,
) -> f32 {
    delta_soh_cycle(p_eff_kw, p_max_kw, t_cell, params.gamma_cycle, dt_hours)
        + delta_soh_calendar(t_cell, soc, params.gamma_cal, params.k_temp, dt_hours)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_power_increases_cycle_degradation() {
        let params = DegradationParams::default();
        let low = delta_soh_cycle(50.0, 500.0, T_REF_KELVIN, params.gamma_cycle, 0.25);
        let high = delta_soh_cycle(500.0, 500.0, T_REF_KELVIN, params.gamma_cycle, 0.25);
        assert!(high > low);
    }

    #[test]
    fn higher_temp_increases_both_terms() {
        let params = DegradationParams::default();
        let cold = total_delta_soh(200.0, 500.0, T_REF_KELVIN, 0.5, &params, 0.25);
        let hot = total_delta_soh(200.0, 500.0, T_REF_KELVIN + 20.0, 0.5, &params, 0.25);
        assert!(hot > cold);
    }

    #[test]
    fn degradation_always_nonnegative() {
        let params = DegradationParams::default();
        let d = total_delta_soh(0.0, 500.0, 250.0, 0.05, &params, 0.25);
        assert!(d >= 0.0);
    }
}
