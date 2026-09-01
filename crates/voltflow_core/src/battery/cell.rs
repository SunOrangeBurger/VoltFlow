//! Coulomb counting & inverter efficiency model (TECHNICAL.md section 4.1).

/// Fixed simulation parameters for the cell/power-conversion model.
#[derive(Debug, Clone, Copy)]
pub struct CellParams {
    pub nominal_energy_kwh: f32, // E_nominal
    pub max_power_kw: f32,       // P_max
    pub inverter_eta: f32,       // eta_inverter
    pub soc_min: f32,
    pub soc_max: f32,
    pub dt_hours: f32, // Delta t
}

impl Default for CellParams {
    fn default() -> Self {
        Self {
            nominal_energy_kwh: 1000.0,
            max_power_kw: 500.0,
            inverter_eta: 0.96,
            soc_min: 0.05,
            soc_max: 0.95,
            dt_hours: 0.25,
        }
    }
}

/// Computes P_eff from the requested action power, applying inverter losses
/// asymmetrically for charge vs. discharge (TECHNICAL.md equation 4.1).
#[inline]
pub fn effective_power_kw(p_action_kw: f32, eta_inverter: f32) -> f32 {
    let eta_safe = eta_inverter.max(1e-6);
    if p_action_kw > 0.0 {
        // Charging: some of the drawn power is lost to inverter conversion.
        p_action_kw * eta_safe
    } else if p_action_kw < 0.0 {
        // Discharging: more must be pulled from the pack to deliver p_action_kw.
        p_action_kw / eta_safe
    } else {
        0.0
    }
}

/// Computes delta SoC for this step given effective power, current SoH.
#[inline]
pub fn delta_soc(p_eff_kw: f32, dt_hours: f32, nominal_energy_kwh: f32, soh: f32) -> f32 {
    let denom = (nominal_energy_kwh * soh).max(1e-6);
    (p_eff_kw * dt_hours) / denom
}

/// Clamps SoC into [soc_min, soc_max].
#[inline]
pub fn clamp_soc(soc: f32, soc_min: f32, soc_max: f32) -> f32 {
    soc.clamp(soc_min, soc_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate2_electrochemical_sanity_1c_1hr() {
        // RESOLVED (see STATUS.md "Known Specification Deviations" for full writeup):
        // Gate 2's prose ("~96%") was checked against the physics, not just
        // the charge-branch arithmetic it happened to match.
        //
        // Physically, inverter conversion always loses energy in BOTH
        // directions:
        //   - Charging: draw P_action from the grid, only P_action*eta
        //     actually reaches the pack (rest lost as inverter heat).
        //     -> multiply. |P_eff| < |P_action|.
        //   - Discharging: to DELIVER P_action to the grid, the pack must
        //     give up MORE than P_action internally (again, inverter loss).
        //     -> divide. |P_eff| > |P_action|.
        //
        // So a 1.0C (1000 kW) discharge request actually pulls ~1041.7 kW
        // from the pack (|dSoC| ~= 104.2%), not 96%. The formula in section
        // 4.1 (multiply on charge, divide on discharge) is physically
        // correct as written; Gate 2's "~96%" prose was simply misquoting
        // the charge-branch number for the discharge case. No code change
        // was needed -- `effective_power_kw` below is confirmed correct.
        let params = CellParams::default();

        let p_eff_discharge = effective_power_kw(-1000.0, params.inverter_eta);
        let dsoc_discharge = delta_soc(p_eff_discharge, 1.0, params.nominal_energy_kwh, 1.0);
        assert!(
            (dsoc_discharge.abs() - 1.041667).abs() < 1e-3,
            "1.0C discharge for 1hr should pull ~104.2% of nominal energy from \
             the pack (more than delivered, due to inverter loss), got {}",
            dsoc_discharge.abs()
        );

        // Charging is the inverse case: draws exactly 1.0C but only 96% of
        // it lands in the pack.
        let p_eff_charge = effective_power_kw(1000.0, params.inverter_eta);
        let dsoc_charge = delta_soc(p_eff_charge, 1.0, params.nominal_energy_kwh, 1.0);
        assert!(
            (dsoc_charge - 0.96).abs() < 1e-4,
            "1.0C charge for 1hr should land ~96% in the pack, got {}",
            dsoc_charge
        );
    }

    #[test]
    fn charging_applies_efficiency_loss_forward() {
        let eff = effective_power_kw(100.0, 0.96);
        assert!((eff - 96.0).abs() < 1e-4);
    }

    #[test]
    fn discharging_applies_efficiency_loss_inverse() {
        let eff = effective_power_kw(-100.0, 0.96);
        assert!((eff - (-104.1667)).abs() < 1e-2);
    }

    #[test]
    fn soc_clamps_to_bounds() {
        assert_eq!(clamp_soc(1.5, 0.05, 0.95), 0.95);
        assert_eq!(clamp_soc(-0.5, 0.05, 0.95), 0.05);
    }
}
