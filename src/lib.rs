#![allow(clippy::needless_doctest_main)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Mathematical optimization and machine learning framework
//! and algorithms.
//!
//! Optimal provides a composable framework
//! for mathematical optimization
//! and machine learning
//! from the optimization perspective,
//! in addition to algorithm implementations.
//!
//! The framework consists of runners,
//! optimizers,
//! and problems,
//! with a chain of dependency as follows:
//! `runner -> optimizer -> problem`.
//! Most optimizers can support many problems
//! and most runners can support many optimizers.
//!
//! A problem defines a mathematical optimization problem.
//! An optimizer defines the steps for solving a problem,
//! usually as an infinite series of state transitions
//! incrementally improving a solution.
//! A runner defines the stopping criteria for an optimizer
//! and may affect the optimization sequence
//! in other ways.
//!
//! # Examples
//!
//! Minimize the "count" problem
//! using a derivative-free optimizer:
//!
//! ```
//! use optimal::binary_argmin;
//!
//! println!(
//!     "{:?}",
//!     binary_argmin(
//!         0,
//!         2,
//!         |point| point.iter().filter(|x| **x).count() as f64,
//!     )
//! );
//! ```
//!
//! Minimize the "sphere" problem
//! using a derivative optimizer:
//!
//! ```
//! use optimal::real_derivative_argmin;
//!
//! println!(
//!     "{:?}",
//!     real_derivative_argmin(
//!         0,
//!         std::iter::repeat(-10.0..=10.0).take(2),
//!         |point| point.iter().map(|x| x.powi(2)).sum(),
//!         |point| point.iter().map(|x| 2.0 * x).collect(),
//!     )
//! );
//! ```
//!
//! For more control over configuration parameters,
//! introspection of the optimization process,
//! serialization,
//! and specialization that may improve performance,
//! see individual optimizer packages.

use std::ops::{Add, Mul, RangeInclusive, Sub};

use optimal_linesearch::{
    backtracking_line_search::{
        BacktrackingLineSearch, BacktrackingRate, SufficientDecreaseParameter,
    },
    initial_step_size::IncrRate,
    step_direction::steepest_descent,
    StepSize,
};
use optimal_pbil::{
    adjust_probabilities, converged, mutate_probabilities, point_from, AdjustRate,
    MutationAdjustRate, MutationChance, NumSamples, Probability, ProbabilityThreshold, Sampleable,
};
use rand::{distributions::Uniform, prelude::*};

/// Return a point that minimizes the given objective function,
/// using a derivative-based optimizer.
///
/// The specific optimizer is subject to change.
///
/// # Arguments
///
/// - `level`: A factor
///   increasing quality of results
///   at the cost of potentially increased time.
///   Values above 0 favor increased quality.
///   Values below 0 favor increased speed.
///   Not all values necessarily have an effect,
///   depending on underlying optimizer.
/// - `initial_bounds`: Bounds for random initial points.
///   Length determines length of points.
/// - `obj_func`: Objective function to minimize.
/// - `obj_func_d`: Derivative of objective function to minimize.
pub fn real_derivative_argmin<F, FD>(
    #[allow(unused_variables)] level: i32,
    initial_bounds: impl IntoIterator<Item = RangeInclusive<f64>>,
    obj_func: F,
    obj_func_d: FD,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    FD: Fn(&[f64]) -> Vec<f64>,
{
    real_derivative_argmin_using(
        level,
        initial_bounds,
        obj_func,
        obj_func_d,
        &mut SmallRng::from_entropy(),
    )
}

/// Return a point that minimizes the given objective function,
/// using a derivative-based optimizer.
///
/// The specific optimizer is subject to change.
///
/// # Arguments
///
/// - `level`: A factor
///   increasing quality of results
///   at the cost of potentially increased time.
///   Values above 0 favor increased quality.
///   Values below 0 favor increased speed.
///   Not all values necessarily have an effect,
///   depending on underlying optimizer.
/// - `initial_bounds`: Bounds for random initial points.
///   Length determines length of points.
/// - `obj_func`: objective function to minimize
/// - `obj_func_d`: derivative of objective function to minimize
/// - `rng`: source of randomness
pub fn real_derivative_argmin_using<F, FD, R>(
    #[allow(unused_variables)] level: i32,
    initial_bounds: impl IntoIterator<Item = RangeInclusive<f64>>,
    obj_func: F,
    obj_func_d: FD,
    rng: &mut R,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
    FD: Fn(&[f64]) -> Vec<f64>,
    R: Rng,
{
    let c_1 = SufficientDecreaseParameter::default();
    let backtracking_rate = BacktrackingRate::default();
    let incr_rate = IncrRate::from_backtracking_rate(backtracking_rate);

    let mut step_size = StepSize::new(1.0).unwrap();
    let mut point = initial_bounds
        .into_iter()
        .map(|range| {
            let (start, end) = range.into_inner();
            Uniform::new_inclusive(start, end).sample(rng)
        })
        .collect::<Vec<_>>();
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

/// Return a point that minimizes the given objective function,
/// using a derivative-based optimizer.
///
/// The specific optimizer is subject to change.
///
/// # Arguments
///
/// - `level`: A factor
///   increasing quality of results
///   at the cost of potentially increased time.
///   Values above 0 favor increased quality.
///   Values below 0 favor increased speed.
///   Not all values necessarily have an effect,
///   depending on underlying optimizer.
/// - `len`: number of elements in each point.
/// - `obj_func`: Objective function to minimize.
pub fn binary_argmin<F>(level: i32, len: usize, obj_func: F) -> Vec<bool>
where
    F: Fn(&[bool]) -> f64,
{
    binary_argmin_using(level, len, obj_func, &mut SmallRng::from_entropy())
}

/// Return a point that minimizes the given objective function,
/// using a derivative-based optimizer.
///
/// The specific optimizer is subject to change.
///
/// # Arguments
///
/// - `level`: A factor
///   increasing quality of results
///   at the cost of potentially increased time.
///   Values above 0 favor increased quality.
///   Values below 0 favor increased speed.
///   Not all values necessarily have an effect,
///   depending on underlying optimizer.
/// - `len`: number of elements in each point.
/// - `obj_func`: objective function to minimize
/// - `rng`: source of randomness
pub fn binary_argmin_using<F, R>(level: i32, len: usize, obj_func: F, rng: &mut R) -> Vec<bool>
where
    F: Fn(&[bool]) -> f64,
    R: Rng,
{
    let (num_samples, adjust_rate, mutation_chance, mutation_adjust_rate, threshold) =
        pbil_config(level, len);

    let mut probabilities = std::iter::repeat(Probability::default())
        .take(len)
        .collect::<Vec<_>>();
    while !converged(threshold, &probabilities) {
        adjust_probabilities(
            adjust_rate,
            &Sampleable::new(&probabilities).best_sample(num_samples, &obj_func, rng),
            &mut probabilities,
        );
        mutate_probabilities(
            &mutation_chance,
            mutation_adjust_rate,
            rng,
            &mut probabilities,
        );
    }
    point_from(&probabilities)
}

fn pbil_config(
    level: i32,
    len: usize,
) -> (
    NumSamples,
    AdjustRate,
    MutationChance,
    MutationAdjustRate,
    ProbabilityThreshold,
) {
    // These numbers are approximate at best.
    let (threshold, num_samples, adjust_rate) = match level {
        x if x > 0 => {
            let x = x as usize + 1;
            (
                ProbabilityThreshold::new(
                    Probability::new(adjust(
                        asymptotic_log_like(1.0, x),
                        ProbabilityThreshold::default().into_inner().into_inner(),
                        0.95,
                    ))
                    .unwrap(),
                )
                .unwrap(),
                NumSamples::new((len * x).min(2)).unwrap(),
                AdjustRate::new(adjust(
                    asymptotic_log_like(1.0, x),
                    AdjustRate::default().into_inner(),
                    0.05,
                ))
                .unwrap(),
            )
        }
        x if x < 0 => {
            let x = x.unsigned_abs() as usize + 1;
            (
                ProbabilityThreshold::new(
                    Probability::new(adjust(
                        asymptotic_log_like(1.0, x),
                        ProbabilityThreshold::default().into_inner().into_inner(),
                        0.55,
                    ))
                    .unwrap(),
                )
                .unwrap(),
                NumSamples::new(
                    ((len as f64 / (1.0 + asymptotic_log_like(1.0, x))) as usize).min(2),
                )
                .unwrap(),
                AdjustRate::new(adjust(
                    asymptotic_log_like(1.0, x),
                    AdjustRate::default().into_inner(),
                    0.5,
                ))
                .unwrap(),
            )
        }
        _ => (
            ProbabilityThreshold::default(),
            NumSamples::new(len.min(2)).unwrap(),
            AdjustRate::default(),
        ),
    };
    (
        num_samples,
        adjust_rate,
        MutationChance::new(0.0).unwrap(),
        MutationAdjustRate::default(),
        threshold,
    )
}

fn asymptotic_log_like(to: f64, from: usize) -> f64 {
    to - to / (from as f64).sqrt()
}

fn adjust<T>(rate: T, x: T, y: T) -> T
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    x + rate * (y - x)
}
