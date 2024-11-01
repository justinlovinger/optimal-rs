use std::hint::black_box;

use computation_types::{arg1, peano::Zero, AnyArg, Computation, ComputationFn, Run};
use optimal_pbil::{PbilBuilder, PbilComputation};
use rand::prelude::*;
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

pub fn pbil_benchmarks() -> impl IntoBenchmarks {
    let len = 20000;

    [benchmark_fn(format!("pbil count {len}"), move |b| {
        b.iter(move || {
            run_pbil(
                black_box(len),
                black_box(arg1!("sample").black_box(count)),
                black_box(SmallRng::seed_from_u64(0)),
            )
        })
    })]
}

fn run_pbil<F, R>(len: usize, obj_func: F, rng: R) -> Vec<bool>
where
    F: Clone + ComputationFn<Dim = Zero, Item = usize>,
    F::Filled: Computation<Dim = Zero, Item = F::Item>,
    R: Rng + AnyArg,
    PbilComputation<F, R>: Run<Output = Vec<bool>>,
{
    PbilBuilder::default()
        .for_(len, obj_func)
        .with(rng)
        .argmin()
}

fn count(point: Vec<bool>) -> usize {
    point.iter().filter(|x| **x).count()
}

tango_benchmarks!(pbil_benchmarks());
tango_main!();
