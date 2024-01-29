use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use optimal_linesearch::{
    backtracking_line_search::BacktrackingLineSearchBuilder, descend,
    step_direction::steepest_descent, StepSize,
};

pub fn bench_line_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("slow");
    group
        .sample_size(50)
        .measurement_time(Duration::from_secs(60))
        .warm_up_time(Duration::from_secs(10))
        .noise_threshold(0.02)
        .significance_level(0.01);

    let len = 100000;
    let initial_point = (1..(len + 1)).map(|x| x as f64).collect::<Vec<_>>();

    group.bench_function(&format!("fixed_step_size skewed_sphere {len}"), |b| {
        b.iter_batched(
            || initial_point.clone(),
            |initial_point| {
                run_fixed_step_size(black_box(skewed_sphere_d), black_box(initial_point))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function(
        &format!("backtracking_line_search skewed_sphere {len}"),
        |b| {
            b.iter_batched(
                || initial_point.clone(),
                |initial_point| {
                    run_backtracking_line_search(
                        black_box(skewed_sphere),
                        black_box(skewed_sphere_d),
                        black_box(initial_point),
                    )
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

pub fn run_fixed_step_size<FD>(obj_func_d: FD, initial_point: Vec<f64>) -> Vec<f64>
where
    FD: Fn(&[f64]) -> Vec<f64>,
{
    let step_size = StepSize::new(0.5).unwrap();
    let mut point = initial_point;
    for _ in 0..2000 {
        point = descend(step_size, &steepest_descent(&obj_func_d(&point)), &point);
    }
    point
}

pub fn run_backtracking_line_search<F, FD>(
    obj_func: F,
    obj_func_d: FD,
    initial_point: Vec<f64>,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Clone + 'static,
    FD: Fn(&[f64]) -> Vec<f64> + 'static,
{
    BacktrackingLineSearchBuilder::default()
        .for_(initial_point.len(), obj_func, obj_func_d)
        .point(initial_point)
        .argmin()
}

fn skewed_sphere(point: &[f64]) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(i, x)| x.powf(1.0 + ((i + 1) as f64) / 100.0))
        .sum()
}

fn skewed_sphere_d(point: &[f64]) -> Vec<f64> {
    point
        .iter()
        .enumerate()
        .map(|(i, x)| (1.0 + ((i + 1) as f64) / 100.0) * x)
        .collect()
}

criterion_group!(benches, bench_line_search);
criterion_main!(benches);
