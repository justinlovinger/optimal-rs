use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use optimal_pbil::*;
use rand::prelude::*;

pub fn bench_pbil(c: &mut Criterion) {
    for len in [10, 100, 1000] {
        c.bench_function(&format!("pbil count {len}"), |b| {
            b.iter_batched(
                || SmallRng::seed_from_u64(0),
                |rng| run_pbil(black_box(count), black_box(len), black_box(rng)),
                BatchSize::SmallInput,
            )
        });
    }
}

fn run_pbil<F, R>(obj_func: F, len: usize, mut rng: R) -> Vec<bool>
where
    F: Fn(&[bool]) -> usize,
    R: Rng,
{
    let num_samples = NumSamples::default();
    let adjust_rate = AdjustRate::default();
    let mutation_chance = MutationChance::default_for(len);
    let mutation_adjust_rate = MutationAdjustRate::default();
    let threshold = ProbabilityThreshold::default();

    let mut probabilities = std::iter::repeat(Probability::default())
        .take(len)
        .collect::<Vec<_>>();
    while !converged(threshold, &probabilities) {
        adjust_probabilities(
            adjust_rate,
            &Sampleable::new(&probabilities).best_sample(num_samples, &obj_func, &mut rng),
            &mut probabilities,
        );
        mutate_probabilities(
            &mutation_chance,
            mutation_adjust_rate,
            &mut rng,
            &mut probabilities,
        );
    }

    point_from(&probabilities)
}

fn count(point: &[bool]) -> usize {
    point.iter().filter(|x| **x).count()
}

criterion_group!(benches, bench_pbil);
criterion_main!(benches);
