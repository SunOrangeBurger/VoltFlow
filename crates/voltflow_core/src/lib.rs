use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod battery;
pub mod data;
pub mod env;

use data::loader::load_market_csv;
use env::simulation::BessSimulation;

#[pyclass]
pub struct RustBessEnv {
    sim: BessSimulation,
}

#[pymethods]
impl RustBessEnv {
    #[new]
    #[pyo3(signature = (csv_path, max_steps, seed=42))]
    pub fn new(csv_path: &str, max_steps: usize, seed: u64) -> PyResult<Self> {
        let market = load_market_csv(csv_path)
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;

        if market.prices.len() < max_steps + 2 {
            return Err(PyRuntimeError::new_err(format!(
                "CSV has only {} rows but max_steps={} requires at least {} rows",
                market.prices.len(),
                max_steps,
                max_steps + 2
            )));
        }

        Ok(Self {
            sim: BessSimulation::new(market, max_steps, seed),
        })
    }

    pub fn reset(&mut self, randomize: bool) -> PyResult<Vec<f32>> {
        Ok(self.sim.reset(randomize).to_vec())
    }

    pub fn step(
        &mut self,
        py: Python<'_>,
        action: f32,
    ) -> PyResult<(Vec<f32>, f32, bool, bool, PyObject)> {
        let (obs, reward, terminated, truncated, info) = self.sim.step(action);

        let info_dict = PyDict::new_bound(py);
        info_dict.set_item("revenue", info.revenue)?;
        info_dict.set_item("degradation_cost", info.degradation_cost)?;
        info_dict.set_item("thermal_penalty", info.thermal_penalty)?;
        info_dict.set_item("soc", info.soc)?;
        info_dict.set_item("soh", info.soh)?;
        info_dict.set_item("t_cell_k", info.t_cell)?;
        info_dict.set_item("price", info.price)?;
        info_dict.set_item("thermal_interlock_active", info.thermal_interlock_active)?;

        Ok((obs.to_vec(), reward, terminated, truncated, info_dict.into()))
    }

    /// Convenience accessor for the current SoC without stepping (used by
    /// the telemetry server for live dashboard state polling).
    pub fn get_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);
        d.set_item("soc", self.sim.soc)?;
        d.set_item("soh", self.sim.soh)?;
        d.set_item("t_cell_k", self.sim.t_cell)?;
        d.set_item("current_step", self.sim.current_step)?;
        Ok(d.into())
    }
}

#[pymodule]
fn voltflow_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBessEnv>()?;
    Ok(())
}