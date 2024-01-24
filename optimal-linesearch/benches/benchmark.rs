use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use optimal_linesearch::{
    backtracking_line_search::{
        BacktrackingLineSearch, BacktrackingRate, SufficientDecreaseParameter,
    },
    descend,
    initial_step_size::IncrRate,
    step_direction::steepest_descent,
    StepSize,
};

pub fn bench_line_search(c: &mut Criterion) {
    for len in [10, 100, 1000] {
        let initial_point = (1..(len + 1)).map(|x| x as f64).collect::<Vec<_>>();
        c.bench_function(&format!("fixed_step_size skewed_sphere {len}"), |b| {
            b.iter_batched(
                || initial_point.clone(),
                |initial_point| {
                    run_fixed_step_size(black_box(skewed_sphere_d), black_box(initial_point))
                },
                BatchSize::SmallInput,
            )
        });
        c.bench_function(
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
    }
}

pub fn run_fixed_step_size<FD>(obj_func_d: FD, initial_point: Vec<f64>) -> Vec<f64>
where
    FD: Fn(&[f64]) -> Vec<f64>,
{
    let step_size = StepSize::new(0.5).unwrap();
    let mut point = initial_point;
    for _ in 0..1000 {
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
    F: Fn(&[f64]) -> f64,
    FD: Fn(&[f64]) -> Vec<f64>,
{
    let c_1 = SufficientDecreaseParameter::default();
    let backtracking_rate = BacktrackingRate::default();
    let incr_rate = IncrRate::from_backtracking_rate(backtracking_rate);

    let mut step_size = StepSize::new(1.0).unwrap();
    let mut point = initial_point;
    for _ in 0..100 {
        let value = obj_func(&point);
        let derivatives = obj_func_d(&point);
        let line_search = BacktrackingLineSearch::new(
            c_1,
            point,
            value,
            &derivatives,
            steepest_descent(&derivatives),
        );
        (step_size, point) = line_search.search(backtracking_rate, &obj_func, step_size);
        step_size = incr_rate * step_size;
    }
    point
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
