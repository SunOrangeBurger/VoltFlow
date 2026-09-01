//! Lumped-capacitance thermal dynamics model (TECHNICAL.md section 4.2).

#[derive(Debug, Clone, Copy)]
pub struct ThermalParams {
    pub v_nominal: f32,      // V_nominal (Volts)
    pub r_internal: f32,     // R_internal (Ohms)
    pub c_thermal: f32,      // C_thermal (J/K)
    pub h_times_a: f32,      // h * A (W/K)
}

impl Default for ThermalParams {
    fn default() -> Self {
        Self {
            v_nominal: 800.0,
            r_internal: 0.015,
            c_thermal: 15_000.0,
            h_times_a: 25.0,
        }
    }
}

/// Heat generation from I^2 * R, where I = P_eff / V_nominal.
/// p_eff_kw is converted to Watts internally for dimensional consistency
/// with V_nominal (Volts) and R_internal (Ohms).
#[inline]
pub fn heat_generated_w(p_eff_kw: f32, v_nominal: f32, r_internal: f32) -> f32 {
    let p_eff_w = p_eff_kw * 1000.0;
    let v_safe = v_nominal.max(1e-6);
    let current = p_eff_w / v_safe;
    current * current * r_internal
}

/// Exact analytic integration of the lumped-capacitance ODE over dt_hours,
/// treating Q_gen as constant across the step (which it is, since it's
/// derived from the step's fixed P_eff).
///
/// The ODE  dT/dt = (Q_gen - hA*(T - T_amb)) / C  has closed-form solution:
///   T(t+dt) = T_amb + Q_gen/hA + (T(t) - T_amb - Q_gen/hA) * exp(-hA/C * dt)
///
/// This is unconditionally stable for any dt (unlike explicit Euler, which
/// is only stable when dt < C/(hA) -- with this model's defaults that's a
/// 600-second time constant, well under a 900-second/15-min step, so plain
/// Euler overshoots and can even cross past ambient in one step). The
/// analytic form costs one extra exp() call and is exact regardless of step
/// size, so there's no accuracy/performance tradeoff here.
#[inline]
pub fn step_temperature(
    t_cell: f32,
    t_ambient: f32,
    q_gen_w: f32,
    params: &ThermalParams,
    dt_hours: f32,
) -> f32 {
    let dt_seconds = dt_hours * 3600.0;
    let h_times_a_safe = params.h_times_a.max(1e-6);
    let c_thermal_safe = params.c_thermal.max(1e-6);

    let steady_state_offset = q_gen_w / h_times_a_safe; // Q_gen/hA
    let decay_rate = h_times_a_safe / c_thermal_safe; // hA/C, units 1/s

    let equilibrium = t_ambient + steady_state_offset;
    equilibrium + (t_cell - equilibrium) * (-decay_rate * dt_seconds).exp()
}

/// Inverts `step_temperature`'s closed-form solution to find the maximum
/// constant heat-generation rate (W) this step can sustain without the
/// *predicted* post-step cell temperature exceeding `t_crit`.
///
/// This is the "predict" half of the hard thermal interlock (STATUS.md
/// deviation #6): rather than only penalizing an over-temperature outcome
/// after the fact via the `kappa*(T-T_crit)^2` reward term, the simulation
/// can call this *before* committing to a requested power level and clamp
/// accordingly -- the same enforcement philosophy `clamp_soc` already
/// applies to the SoC boundary.
///
/// From `step_temperature`'s equilibrium form,
///   T_new = T_amb*(1-k) + k*T_cell + (Q/hA)*(1-k),  where k = exp(-hA/C * dt)
/// solving for Q at T_new = t_crit gives the formula below. Returns 0.0
/// (never negative) if even zero heat generation would already predict a
/// temperature at or above `t_crit` this step -- resistive heating
/// (`heat_generated_w`) can't be negative, so a power clamp alone can't
/// pull the prediction back under `t_crit` in that case; zero is the best
/// a power-only interlock can do, and any residual overshoot is a
/// pre-existing thermal condition, not something this step's action
/// caused.
#[inline]
pub fn max_q_gen_w_for_t_crit(
    t_cell: f32,
    t_ambient: f32,
    params: &ThermalParams,
    dt_hours: f32,
    t_crit: f32,
) -> f32 {
    let dt_seconds = dt_hours * 3600.0;
    let h_times_a_safe = params.h_times_a.max(1e-6);
    let c_thermal_safe = params.c_thermal.max(1e-6);
    let decay_rate = h_times_a_safe / c_thermal_safe;
    let k = (-decay_rate * dt_seconds).exp();
    // Guard against (1-k) collapsing to ~0 (e.g. dt_hours == 0, or a
    // pathologically slow decay rate), which would blow the division up.
    let one_minus_k = (1.0 - k).max(1e-6);

    let q_max = h_times_a_safe * (t_crit - t_ambient * (1.0 - k) - k * t_cell) / one_minus_k;
    q_max.max(0.0)
}

/// Inverts `heat_generated_w` (Q = I^2*R, I = P_eff/V) to find the maximum
/// |P_eff| (kW) that stays at or under a given heat-generation ceiling.
/// Heat scales with current squared, so this is direction-agnostic: it
/// returns a magnitude, and the caller re-applies the original sign of the
/// requested P_eff (charge vs. discharge).
#[inline]
pub fn max_p_eff_kw_for_heat_limit(q_max_w: f32, v_nominal: f32, r_internal: f32) -> f32 {
    if q_max_w <= 0.0 {
        return 0.0;
    }
    let v_safe = v_nominal.max(1e-6);
    let r_safe = r_internal.max(1e-6);
    let current_max = (q_max_w / r_safe).sqrt();
    (current_max * v_safe) / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_power_cools_toward_ambient() {
        let params = ThermalParams::default();
        let t_new = step_temperature(310.0, 298.15, 0.0, &params, 0.25);
        assert!(t_new < 310.0);
        assert!(t_new > 298.15);
    }

    #[test]
    fn large_timestep_does_not_overshoot_ambient() {
        // Regression test: with this model's default params, the thermal
        // time constant is C/(hA) = 15000/25 = 600 seconds, which is LESS
        // than a single 15-min (900s) simulation step. Explicit Euler
        // integration overshoots past ambient in this regime (previously
        // observed: 310K -> 292.2K, undershooting below the 298.15K
        // ambient it was cooling toward). The analytic solution must never
        // overshoot regardless of step size.
        let params = ThermalParams::default();
        let t_new = step_temperature(310.0, 298.15, 0.0, &params, 0.25);
        assert!(
            t_new >= 298.15,
            "cooling toward ambient must not cross past it in one step, got {}",
            t_new
        );

        // Even with a much larger step, should still land strictly between
        // start and ambient, converging monotonically -- never oscillating.
        let t_new_big_step = step_temperature(310.0, 298.15, 0.0, &params, 2.0);
        assert!(t_new_big_step >= 298.15 && t_new_big_step < 310.0);
    }

    #[test]
    fn heat_generation_pushes_above_ambient_at_equilibrium() {
        let params = ThermalParams::default();
        // With constant heat generation, temperature should rise above
        // ambient rather than settle back to it.
        let t_new = step_temperature(298.15, 298.15, 5000.0, &params, 0.25);
        assert!(t_new > 298.15);
    }

    #[test]
    fn heat_generation_scales_with_current_squared() {
        let q1 = heat_generated_w(100.0, 800.0, 0.015);
        let q2 = heat_generated_w(200.0, 800.0, 0.015);
        // Doubling power should quadruple heat (I^2 relationship).
        assert!((q2 / q1 - 4.0).abs() < 1e-2);
    }

    // --- Hard thermal interlock: predict-then-clamp (STATUS.md deviation #6) ---

    #[test]
    fn max_q_gen_round_trips_through_step_temperature() {
        // The whole point of the inversion: feeding max_q_gen_w_for_t_crit's
        // output straight back into step_temperature should land (very
        // close to) exactly at t_crit, not above or wildly under it.
        let params = ThermalParams::default();
        let t_cell = 310.0;
        let t_ambient = 295.0;
        let t_crit = 318.15;
        let dt_hours = 0.25;

        let q_max = max_q_gen_w_for_t_crit(t_cell, t_ambient, &params, dt_hours, t_crit);
        let t_new = step_temperature(t_cell, t_ambient, q_max, &params, dt_hours);
        assert!(
            (t_new - t_crit).abs() < 1e-2,
            "expected predicted temp to land at t_crit ({t_crit}), got {t_new}"
        );
    }

    #[test]
    fn max_q_gen_is_zero_when_ambient_alone_would_exceed_t_crit() {
        // Zero heat budget is only correct when even *natural relaxation*
        // (q=0) can't bring the predicted temp under t_crit -- i.e. both
        // t_cell and t_ambient already sit above t_crit, so their weighted
        // average (the q=0 case) must stay above t_crit too. (If ambient
        // were below t_crit, q=0 could still cool the cell under t_crit by
        // end of step even starting above it -- that's a legitimate
        // positive budget, not a bug; see the round-trip test above.)
        let params = ThermalParams::default();
        let q_max = max_q_gen_w_for_t_crit(320.0, 319.0, &params, 0.25, 318.15);
        assert_eq!(q_max, 0.0);
    }

    #[test]
    fn max_q_gen_can_be_positive_even_when_t_cell_exceeds_t_crit() {
        // Starting above t_crit doesn't automatically mean zero budget: if
        // ambient is comfortably below t_crit, natural relaxation this step
        // can still land the prediction under t_crit, leaving positive
        // headroom for some heat generation.
        let params = ThermalParams::default();
        let q_max = max_q_gen_w_for_t_crit(320.0, 295.0, &params, 0.25, 318.15);
        assert!(q_max > 0.0, "expected positive headroom, got {q_max}");
        // And it must still round-trip to (approximately) t_crit exactly.
        let t_new = step_temperature(320.0, 295.0, q_max, &params, 0.25);
        assert!((t_new - 318.15).abs() < 1e-2);
    }

    #[test]
    fn max_q_gen_never_negative_or_nan() {
        let params = ThermalParams::default();
        // Sweep a range of starting temps, including pathological ones
        // above t_crit and with dt_hours = 0.
        for &t_cell in &[200.0, 300.0, 318.15, 340.0, 373.15] {
            for &dt in &[0.0, 0.25, 2.0] {
                let q = max_q_gen_w_for_t_crit(t_cell, 295.0, &params, dt, 318.15);
                assert!(q.is_finite() && q >= 0.0, "t_cell={t_cell} dt={dt} -> q={q}");
            }
        }
    }

    #[test]
    fn max_p_eff_inverts_heat_generated_w() {
        let v = 800.0;
        let r = 0.015;
        let p_eff_kw = 350.0;
        let q = heat_generated_w(p_eff_kw, v, r);
        let p_eff_recovered = max_p_eff_kw_for_heat_limit(q, v, r);
        assert!(
            (p_eff_recovered - p_eff_kw).abs() < 1e-1,
            "expected round-trip to recover ~{p_eff_kw}, got {p_eff_recovered}"
        );
    }

    #[test]
    fn max_p_eff_is_zero_for_zero_or_negative_budget() {
        assert_eq!(max_p_eff_kw_for_heat_limit(0.0, 800.0, 0.015), 0.0);
        assert_eq!(max_p_eff_kw_for_heat_limit(-5.0, 800.0, 0.015), 0.0);
    }
}