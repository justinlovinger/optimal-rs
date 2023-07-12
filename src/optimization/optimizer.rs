use blanket::blanket;

use crate::prelude::*;

/// Running optimizer methods
/// independent of configuration
/// and state.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Optimizer<P>
where
    P: Problem,
{
    /// Return the best point discovered.
    fn best_point(&self) -> P::Point<'_>;

    /// Return the value of the best point discovered,
    /// evaluating the best point
    /// if necessary.
    fn best_point_value(&self) -> P::Value;
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use super::*;

    assert_obj_safe!(Optimizer<()>);
}
