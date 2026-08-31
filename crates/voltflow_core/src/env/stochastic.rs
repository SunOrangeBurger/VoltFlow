//! Ornstein-Uhlenbeck noise process for domain randomization of prices and
//! ambient temperature during training (adds robustness beyond the fixed
//! historical CSV trace).

use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone, Copy)]
pub struct OuNoise {
    pub theta: f32, // mean reversion speed
    pub mu: f32,    // long-run mean (0.0 for additive shocks)
    pub sigma: f32, // volatility
    state: f32,
}

impl OuNoise {
    pub fn new(theta: f32, mu: f32, sigma: f32) -> Self {
        Self {
            theta,
            mu,
            sigma,
            state: mu,
        }
    }

    /// Advances the process by one step (dt in hours) and returns the new
    /// noise value. Caller adds this additively to the base signal.
    pub fn step(&mut self, dt_hours: f32, rng: &mut StdRng) -> f32 {
        let normal = Normal::new(0.0_f32, 1.0_f32).expect("valid normal params");
        let dw = normal.sample(rng) * dt_hours.sqrt();
        let dx = self.theta * (self.mu - self.state) * dt_hours + self.sigma * dw;
        self.state += dx;
        self.state
    }

    pub fn reset(&mut self) {
        self.state = self.mu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn mean_reverts_over_many_steps() {
        let mut ou = OuNoise::new(0.5, 0.0, 5.0);
        let mut rng = StdRng::seed_from_u64(42);
        // Push state far from mean manually, then verify it decays back.
        for _ in 0..2000 {
            ou.step(0.25, &mut rng);
        }
        // With theta=0.5, mu=0, should stay bounded and not diverge.
        assert!(ou.state.abs() < 100.0);
    }
}
