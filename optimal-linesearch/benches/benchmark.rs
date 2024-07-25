use std::hint::black_box;

use optimal_linesearch::{
    backtracking_line_search::{
        BacktrackingLineSearchBuilder, BacktrackingLineSearchStoppingCriteria, BfgsInitializer,
        StepDirection,
    },
    descend,
    step_direction::steepest_descent,
    StepSize,
};
use tango_bench::{benchmark_fn, tango_benchmarks, tango_main, IntoBenchmarks};

pub fn linesearch_benchmarks() -> impl IntoBenchmarks {
    let len = 100000;

    // BFGS requires quadratic time and memory,
    // relative to the length of points.
    let bfgs_len = 400;

    [
        benchmark_fn(
            format!("fixed_step_size steepest skewed_sphere {len}"),
            move |b| {
                b.iter(move || {
                    run_fixed_step_size(
                        black_box(skewed_sphere_d),
                        black_box(skewed_sphere_initial_point(len)),
                    )
                })
            },
        ),
        benchmark_fn(
            format!("backtracking_line_search steepest skewed_sphere {len}"),
            move |b| {
                b.iter(move || {
                    run_backtracking_line_search(
                        StepDirection::Steepest,
                        black_box(skewed_sphere),
                        black_box(skewed_sphere_d),
                        black_box(skewed_sphere_initial_point(len)),
                    )
                })
            },
        ),
        benchmark_fn(
            format!("backtracking_line_search bfgs skewed_sphere {len}"),
            move |b| {
                b.iter(move || {
                    run_backtracking_line_search(
                        StepDirection::Bfgs {
                            initializer: BfgsInitializer::Gamma,
                        },
                        black_box(skewed_sphere),
                        black_box(skewed_sphere_d),
                        black_box(skewed_sphere_initial_point(bfgs_len)),
                    )
                })
            },
        ),
    ]
}

fn skewed_sphere_initial_point(len: usize) -> Vec<f64> {
    (1..(len + 1)).map(|x| x as f64).collect()
}

pub fn run_fixed_step_size<FD>(obj_func_d: FD, initial_point: Vec<f64>) -> Vec<f64>
where
    FD: Fn(&[f64]) -> Vec<f64>,
{
    let step_size = StepSize::new(0.5).unwrap();
    let mut point = initial_point;
    for _ in 0..2000 {
        point = descend(step_size, steepest_descent(obj_func_d(&point)), point).collect();
    }
    point
}

pub fn run_backtracking_line_search<F, FD>(
    step_direction: StepDirection,
    obj_func: F,
    obj_func_d: FD,
    initial_point: Vec<f64>,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Clone + 'static,
    FD: Fn(&[f64]) -> Vec<f64> + 'static,
{
    BacktrackingLineSearchBuilder::default()
        .direction(step_direction)
        .stopping_criteria(BacktrackingLineSearchStoppingCriteria::Iteration(100))
        .for_(initial_point.len(), obj_func, obj_func_d)
        .with_point(initial_point)
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

tango_benchmarks!(linesearch_benchmarks());
tango_main!();
