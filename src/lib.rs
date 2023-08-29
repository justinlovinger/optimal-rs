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
//! use optimal::{prelude::*, BinaryDerivativeFreeConfig};
//!
//! println!(
//!     "{:?}",
//!     BinaryDerivativeFreeConfig::start_default_for(16, |point| {
//!         point.iter().filter(|x| **x).count() as f64
//!     })
//!     .argmin()
//! );
//! ```
//!
//! Minimize the "sphere" problem
//! using a derivative optimizer:
//!
//! ```
//! use optimal::{prelude::*, RealDerivativeConfig};
//!
//! println!(
//!     "{:?}",
//!     RealDerivativeConfig::start_default_for(
//!         2,
//!         std::iter::repeat(-10.0..=10.0).take(2),
//!         |point| point.iter().map(|x| x.powi(2)).sum(),
//!         |point| point.iter().map(|x| 2.0 * x).collect(),
//!     )
//!     .nth(100)
//!     .unwrap()
//!     .best_point()
//! );
//! ```
//!
//! For more control over configuration parameters,
//! introspection of the optimization process,
//! serialization,
//! and specialization that may improve performance,
//! see individual optimizer packages.

use std::{
    fmt::Debug,
    ops::{Add, Mul, RangeInclusive, Sub},
};

pub use optimal_core::prelude;
pub use optimal_core::prelude::*;
use optimal_pbil::{
    AdjustRate, MutationAdjustRate, MutationChance, NumSamples, Pbil, Probability,
    ProbabilityThreshold, UntilProbabilitiesConverged, UntilProbabilitiesConvergedConfig,
};
use optimal_steepest::backtracking_steepest::{self, BacktrackingSteepest};
use rand_xoshiro::SplitMix64;

/// A generic binary derivative-free optimizer.
#[allow(missing_debug_implementations)]
pub struct RealDerivative {
    // This should wrap a runner
    // so `argmin` and the like can be used,
    // but as of 2023-08-23,
    // we lack an appropriate runner.
    // A derivative norm stopping criteria would be appropriate.
    #[allow(clippy::type_complexity)]
    inner: BacktrackingSteepest<
        f64,
        Box<dyn Fn(&[f64]) -> f64>,
        Box<dyn Fn(&[f64]) -> (f64, Vec<f64>)>,
    >,
}

/// Binary derivative-free optimizer configuration parameters.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct RealDerivativeConfig {
    /// A scaling factor
    /// increasing quality of results
    /// at the cost of potentially increased time.
    /// Values above 0 favor increased optimality.
    /// Values below 0 favor increased speed.
    pub optimality: i32,
}

impl StreamingIterator for RealDerivative {
    type Item = Self;

    fn advance(&mut self) {
        self.inner.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
}

impl Optimizer for RealDerivative {
    type Point = Vec<f64>;

    fn best_point(&self) -> Self::Point {
        self.inner.best_point()
    }
}

impl RealDerivativeConfig {
    /// Return this optimizer default
    /// running on the given problem.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `initial_bounds`: bounds for random initial points
    /// - `obj_func`: objective function to minimize
    /// - `obj_func_d`: derivative of objective function to minimize
    pub fn start_default_for<F, FD>(
        len: usize,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<f64>>,
        obj_func: F,
        obj_func_d: FD,
    ) -> RealDerivative
    where
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        Self::default().start(len, initial_bounds, obj_func, obj_func_d)
    }

    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `initial_bounds`: bounds for random initial points
    /// - `obj_func`: objective function to minimize
    /// - `obj_func_d`: derivative of objective function to minimize
    pub fn start<F, FD>(
        self,
        len: usize,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<f64>>,
        obj_func: F,
        obj_func_d: FD,
    ) -> RealDerivative
    where
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        RealDerivative {
            inner: self.inner_config(len).start(
                initial_bounds,
                Box::new(obj_func.clone()),
                Box::new(move |x| (obj_func(x), obj_func_d(x))),
            ),
        }
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `initial_bounds`: bounds for random initial points
    /// - `obj_func`: objective function to minimize
    /// - `obj_func_d`: derivative of objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<B, F, FD>(
        self,
        len: usize,
        initial_bounds: impl IntoIterator<Item = RangeInclusive<f64>>,
        obj_func: F,
        obj_func_d: FD,
        rng: &mut SplitMix64,
    ) -> RealDerivative
    where
        F: Fn(&[f64]) -> f64 + Clone + 'static,
        FD: Fn(&[f64]) -> Vec<f64> + 'static,
    {
        RealDerivative {
            inner: self.inner_config(len).start_using(
                initial_bounds,
                Box::new(obj_func.clone()),
                Box::new(move |x| (obj_func(x), obj_func_d(x))),
                rng,
            ),
        }
    }

    fn inner_config(self, _len: usize) -> backtracking_steepest::Config<f64> {
        // With line search,
        // backtracking steepest is always optimal,
        // regardless of parameters.
        backtracking_steepest::Config::default()
    }
}

/// A generic binary derivative-free optimizer.
#[allow(missing_debug_implementations)]
pub struct BinaryDerivativeFree {
    #[allow(clippy::type_complexity)]
    inner: UntilProbabilitiesConverged<Pbil<f64, Box<dyn Fn(&[bool]) -> f64>>>,
}

/// Binary derivative-free optimizer configuration parameters.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BinaryDerivativeFreeConfig {
    /// A scaling factor
    /// increasing quality of results
    /// at the cost of potentially increased time.
    /// Values above 0 favor increased optimality.
    /// Values below 0 favor increased speed.
    pub optimality: i32,
}

impl StreamingIterator for BinaryDerivativeFree {
    type Item = Self;

    fn advance(&mut self) {
        self.inner.advance()
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
}

impl Optimizer for BinaryDerivativeFree {
    type Point = Vec<bool>;

    fn best_point(&self) -> Self::Point {
        self.inner.it().best_point()
    }
}

impl BinaryDerivativeFreeConfig {
    /// Return this optimizer default
    /// running on the given problem.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `obj_func`: objective function to minimize
    pub fn start_default_for<F>(len: usize, obj_func: F) -> BinaryDerivativeFree
    where
        F: Fn(&[bool]) -> f64 + 'static,
    {
        Self::default().start(len, obj_func)
    }

    /// Return this optimizer
    /// running on the given problem.
    ///
    /// This may be nondeterministic.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `obj_func`: objective function to minimize
    pub fn start<F>(self, len: usize, obj_func: F) -> BinaryDerivativeFree
    where
        F: Fn(&[bool]) -> f64 + 'static,
    {
        let (runner_config, config) = self.inner_config(len);
        BinaryDerivativeFree {
            inner: runner_config.start(config.start(len, Box::new(obj_func))),
        }
    }

    /// Return this optimizer
    /// running on the given problem
    /// initialized using `rng`.
    ///
    /// # Arguments
    ///
    /// - `len`: number of elements in each point
    /// - `obj_func`: objective function to minimize
    /// - `rng`: source of randomness
    pub fn start_using<B, F>(
        self,
        len: usize,
        obj_func: F,
        rng: &mut SplitMix64,
    ) -> BinaryDerivativeFree
    where
        F: Fn(&[bool]) -> f64 + 'static,
    {
        let (runner_config, config) = self.inner_config(len);
        BinaryDerivativeFree {
            inner: runner_config.start(config.start_using(len, Box::new(obj_func), rng)),
        }
    }

    fn inner_config(self, len: usize) -> (UntilProbabilitiesConvergedConfig, optimal_pbil::Config) {
        // These numbers are approximate at best.
        let (threshold, num_samples, adjust_rate) = match self.optimality {
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
            UntilProbabilitiesConvergedConfig { threshold },
            optimal_pbil::Config {
                num_samples,
                adjust_rate,
                mutation_chance: MutationChance::new(0.0).unwrap(),
                mutation_adjust_rate: MutationAdjustRate::default(),
            },
        )
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
