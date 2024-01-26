use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use optimal_pbil::*;
use rand::prelude::*;

pub fn bench_pbil(c: &mut Criterion) {
    for len in [10, 100, 1000] {
        c.bench_function(&format!("pbil count {len}"), |b| {
            b.iter_batched(
                || SmallRng::seed_from_u64(0),
                |rng| run_pbil(black_box(len), black_box(count), black_box(rng)),
                BatchSize::SmallInput,
            )
        });
    }
}

fn run_pbil<F, R>(len: usize, obj_func: F, rng: R) -> Vec<bool>
where
    F: Fn(&[bool]) -> usize,
    R: Rng,
{
    PbilBuilder::default()
        .for_(len, obj_func)
        .with(rng)
        .argmin()
}

fn count(point: &[bool]) -> usize {
    point.iter().filter(|x| **x).count()
}

criterion_group!(benches, bench_pbil);
criterion_main!(benches);
