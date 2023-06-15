//! Useful traits, types, and functions unlikely to conflict with existing definitions.

pub use streaming_iterator::StreamingIterator;

pub use crate::{
    optimization::{config::*, optimizer::*, runner::*},
    optimizer::derivative::StepSize,
    problem::*,
    traits::DefaultFor,
};
