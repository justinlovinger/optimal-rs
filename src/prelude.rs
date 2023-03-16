//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use crate::{
    optimizer::{
        derivative::StepSize, BestPoint, BestPointValue, Initialize, InitializeUsing,
        IntoStreamingIterator, IsDone, Point, Points, Step,
    },
    problem::*,
};
