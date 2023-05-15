//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use streaming_iterator::StreamingIterator;

pub use crate::{
    optimizer::{
        derivative::StepSize, Convergent, Optimizer, OptimizerArgmin, OptimizerConfig,
        OptimizerConfigless, OptimizerProblem, OptimizerState, PointBased, PopulationBased,
        RunningOptimizer, RunningOptimizerConfigless, StochasticOptimizerConfig,
    },
    problem::*,
    traits::DefaultFor,
};
