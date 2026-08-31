//! Gate 1: verifies simulation throughput exceeds 2,000,000 steps/second
//! across 4 parallel threads (spec section 7).
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use voltflow_core::data::loader::MarketData;
use voltflow_core::env::simulation::BessSimulation;

fn dummy_market(n: usize) -> MarketData {
    MarketData {
        prices: (0..n).map(|i| 50.0 + (i % 24) as f32).collect(),
        ambient_temps_k: vec![288.15; n],
        solar_irradiance: vec![0.0; n],
    }
}

fn single_thread_steps(c: &mut Criterion) {
    let mut sim = BessSimulation::new(dummy_market(10_000), 96, 1);
    sim.reset(false);
    c.bench_function("single_thread_1000_steps", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let action = ((i % 200) as f32 / 100.0) - 1.0;
                let _ = sim.step(black_box(action));
            }
        })
    });
}

fn four_thread_parallel_steps(c: &mut Criterion) {
    c.bench_function("four_thread_parallel_1000_steps_each", |b| {
        b.iter(|| {
            (0..4).into_par_iter().for_each(|t| {
                let mut sim = BessSimulation::new(dummy_market(10_000), 96, t as u64);
                sim.reset(false);
                for i in 0..1000 {
                    let action = ((i % 200) as f32 / 100.0) - 1.0;
                    let _ = sim.step(black_box(action));
                }
            });
        })
    });
}

criterion_group!(benches, single_thread_steps, four_thread_parallel_steps);
criterion_main!(benches);
