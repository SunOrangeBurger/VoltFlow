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
}