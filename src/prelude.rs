//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use crate::{
    optimizer::{
        derivative::StepSize, Convergent, IntoStreamingIterator, OptimizerConfig, PointBased,
        PopulationBased, RunningOptimizer, StochasticOptimizerConfig,
    },
    problem::*,
};
