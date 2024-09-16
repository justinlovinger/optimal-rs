#![allow(clippy::needless_doctest_main)]
#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

//! Mathematical optimization and machine-learning components and algorithms.
//!
//! Optimal provides a framework
//! for mathematical optimization
//! and machine-learning
//! from the optimization-perspective.
//!
//! This package provides a relatively stable high-level API
//! for broadly defined optimizers.
//!
//! For more control over configuration,
//! introspection of the optimization process,
//! serialization,
//! and specialization that may improve performance,
//! see individual optimizer packages.
//!
//! For the computation-framework powering Optimal,
//! see `optimal-compute-core`.
//!
//! Note: more specific support for machine-learning will be added in the future.
//! Currently,
//! machine-learning is supported
//! by defining an objective-function
//! taking model-parameters.
//!
//! # Examples
//!
//! Minimize the "count" problem
//! using a derivative-free optimizer:
//!
//! ```
//! use optimal::Binary;
//!
//! println!(
//!     "{:?}",
//!     Binary::default()
//!         .for_(2, |point| point.iter().filter(|x| **x).count() as f64)
//!         .argmin()
//! );
//! ```
//!
//! Minimize the "sphere" problem
//! using a derivative optimizer:
//!
//! ```
//! use optimal::RealDerivative;
//!
//! println!(
//!     "{:?}",
//!     RealDerivative::default()
//!         .for_(
//!             std::iter::repeat(-10.0..=10.0).take(2),
//!             |point| point.iter().map(|x| x.powi(2)).sum(),
//!             |point| point.iter().map(|x| 2.0 * x).collect()
//!         )
//!         .argmin()
//! );
//! ```

use std::ops::{Add, Mul, RangeInclusive, Sub};

use optimal_compute_core::{arg1, peano::Zero, run::Value, Computation};
use optimal_linesearch::backtracking_line_search::BacktrackingLineSearchBuilder;
use optimal_pbil::{types::*, Pbil, PbilStoppingCriteria};
use rand::{distributions::Uniform, prelude::*};

/// An optimizer for real-derivative problems.
///
/// The specific optimizer is subject to change.
#[derive(Clone, Debug, Default)]
pub struct RealDerivative {
    /// A factor
    /// increasing quality of results
    /// at the cost of potentially increased time.
    /// Values above 0 favor increased quality.
    /// Values below 0 favor increased speed.
    /// Not all values necessarily have an effect,
    /// depending on underlying optimizer.
    pub level: i32,
}

impl RealDerivative {
    /// Prepare for a specific problem.
    pub fn for_<I, F, FD>(
        self,
        initial_bounds: I,
        obj_func: F,
        obj_func_d: FD,
    ) -> RealDerivativeFor<I, F, FD>
    where
        I: IntoIterator<Item = RangeInclusive<f64>>,
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        RealDerivativeFor {
            agnostic: self,
            initial_bounds,
            obj_func,
            obj_func_d,
        }
    }
}

/// An optimizer for a specific real-derivative problem.
#[derive(Clone, Debug)]
pub struct RealDerivativeFor<I, F, FD> {
    #[allow(dead_code)]
    agnostic: RealDerivative,
    /// `initial_bounds`: Bounds for random initial points.
    /// Length determines length of points.
    initial_bounds: I,
    /// Objective function to minimize.
    obj_func: F,
    /// Derivative of objective function to minimize.
    obj_func_d: FD,
}

impl<I, F, FD> RealDerivativeFor<I, F, FD> {
    /// Use a specific source of randomness.
    pub fn with<R>(self, rng: R) -> RealDerivativeWith<I, F, FD, R>
    where
        R: Rng,
    {
        RealDerivativeWith { problem: self, rng }
    }

    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<f64>
    where
        I: IntoIterator<Item = RangeInclusive<f64>>,
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        self.with(SmallRng::from_entropy()).argmin()
    }
}

/// An optimizer for a specific real-derivative problem
/// with a specific source of randomness.
#[derive(Clone, Debug)]
pub struct RealDerivativeWith<I, F, FD, R> {
    problem: RealDerivativeFor<I, F, FD>,
    /// Source of randomness.
    rng: R,
}

impl<I, F, FD, R> RealDerivativeWith<I, F, FD, R> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(mut self) -> Vec<f64>
    where
        I: IntoIterator<Item = RangeInclusive<f64>>,
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
        R: Rng,
    {
        let initial_point = self
            .problem
            .initial_bounds
            .into_iter()
            .map(|range| {
                let (start, end) = range.into_inner();
                Uniform::new_inclusive(start, end).sample(&mut self.rng)
            })
            .collect::<Vec<_>>();

        BacktrackingLineSearchBuilder::default()
            .for_combined(
                initial_point.len(),
                |point| Value((self.problem.obj_func)(&point)),
                |point| {
                    (
                        Value((self.problem.obj_func)(&point)),
                        Value((self.problem.obj_func_d)(&point)),
                    )
                },
            )
            .with_point(initial_point)
            .argmin()
    }
}

/// An optimizer for binary problems.
///
/// The specific optimizer is subject to change.
#[derive(Clone, Debug, Default)]
pub struct Binary {
    /// A factor
    /// increasing quality of results
    /// at the cost of potentially increased time.
    /// Values above 0 favor increased quality.
    /// Values below 0 favor increased speed.
    /// Not all values necessarily have an effect,
    /// depending on underlying optimizer.
    pub level: i32,
}

impl Binary {
    /// Prepare for a specific problem.
    pub fn for_<F>(self, len: usize, obj_func: F) -> BinaryFor<F>
    where
        F: Fn(&[bool]) -> f64,
    {
        BinaryFor {
            agnostic: self,
            len,
            obj_func,
        }
    }
}

/// An optimizer for a specific binary problem.
#[derive(Clone, Debug)]
pub struct BinaryFor<F> {
    agnostic: Binary,
    /// number of elements in each point.
    len: usize,
    /// Objective function to minimize.
    obj_func: F,
}

impl<F> BinaryFor<F> {
    /// Use a specific seed for randomness.
    pub fn with(self, seed: u64) -> BinaryWith<F> {
        BinaryWith {
            problem: self,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> f64,
    {
        BinaryWith {
            problem: self,
            rng: SmallRng::from_entropy(),
        }
        .argmin()
    }
}

/// An optimizer for a specific binary problem
/// with a specific source of randomness.
#[derive(Clone, Debug)]
pub struct BinaryWith<F> {
    problem: BinaryFor<F>,
    rng: SmallRng,
}

impl<F> BinaryWith<F> {
    /// Return a point that attempts to minimize the given objective function.
    pub fn argmin(self) -> Vec<bool>
    where
        F: Fn(&[bool]) -> f64,
    {
        pbil_config(self.problem.agnostic.level, self.problem.len)
            .for_(
                self.problem.len,
                arg1!("sample").black_box::<_, Zero, usize>(|sample: Vec<bool>| {
                    Value((self.problem.obj_func)(&sample))
                }),
            )
            .with(self.rng)
            .argmin()
    }
}

fn pbil_config(level: i32, len: usize) -> Pbil {
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
    Pbil {
        num_samples,
        adjust_rate,
        mutation_chance: MutationChance::new(0.0).unwrap(),
        mutation_adjust_rate: MutationAdjustRate::default(),
        stopping_criteria: PbilStoppingCriteria::Threshold(threshold),
    }
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
