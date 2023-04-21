//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use crate::{
    optimizer::{
        derivative::StepSize, Convergent, IntoStreamingIterator, Optimizer, OptimizerBase,
        OptimizerConfig, OptimizerDeinitialization, OptimizerStep, PointBased, PopulationBased,
        RunningOptimizerExt, StochasticOptimizerConfig,
    },
    problem::*,
};
