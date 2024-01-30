use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use optimal_pbil::PbilBuilder;
use rand::prelude::*;

pub fn bench_pbil(c: &mut Criterion) {
    let mut group = c.benchmark_group("slow");
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(60))
        .warm_up_time(Duration::from_secs(10))
        .noise_threshold(0.02)
        .significance_level(0.01);

    let len = 20000;

    group.bench_function(&format!("pbil count {len}"), |b| {
        b.iter_batched(
            || SmallRng::seed_from_u64(0),
            |rng| run_pbil(black_box(len), black_box(count), black_box(rng)),
            BatchSize::SmallInput,
        )
    });
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
