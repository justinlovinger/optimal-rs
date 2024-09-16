use std::hint::black_box;

use optimal_compute_core::{arg1, peano::Zero, run::Value, Computation, ComputationFn, Run};
use optimal_pbil::{PbilBuilder, PbilComputation};
use rand::prelude::*;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

pub fn pbil_benchmarks() -> impl IntoBenchmarks {
    let len = 20000;

    [benchmark_fn(format!("pbil count {len}"), move |b| {
        b.iter(move || {
            run_pbil(
                black_box(len),
                black_box(arg1!("sample").black_box::<_, Zero, usize>(count)),
                black_box(SmallRng::seed_from_u64(0)),
            )
        })
    })]
}

fn run_pbil<F, R>(len: usize, obj_func: F, rng: R) -> Vec<bool>
where
    F: ComputationFn<Dim = Zero, Item = usize>,
    R: Rng,
    PbilComputation<F, R>: Run<Output = Vec<bool>>,
{
    PbilBuilder::default()
        .for_(len, obj_func)
        .with(rng)
        .argmin()
}

fn count(point: Vec<bool>) -> Value<usize> {
    Value(point.iter().filter(|x| **x).count())
}

tango_benchmarks!(pbil_benchmarks());
tango_main!();
