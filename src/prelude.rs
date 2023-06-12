//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use streaming_iterator::StreamingIterator;

pub use crate::{
    optimization::{
        Convergent, Optimizer, OptimizerArgmin, OptimizerConfig, OptimizerConfigless,
        OptimizerProblem, OptimizerState, RunningOptimizer, RunningOptimizerConfigless,
        StochasticOptimizerConfig,
    },
    optimizer::derivative::StepSize,
    problem::*,
    traits::DefaultFor,
};
