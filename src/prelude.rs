//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use streaming_iterator::StreamingIterator;

pub use crate::{
    optimization::{
        config::{Convergent, OptimizerConfig, OptimizerState, StochasticOptimizerConfig},
        optimizer::{
            Optimizer, OptimizerArgmin, OptimizerConfigless, OptimizerProblem, RunningOptimizer,
            RunningOptimizerConfigless,
        },
    },
    optimizer::derivative::StepSize,
    problem::*,
    traits::DefaultFor,
};
