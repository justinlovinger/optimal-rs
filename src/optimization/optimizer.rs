use blanket::blanket;

/// Running optimizer methods
/// independent of configuration
/// and state.
#[blanket(derive(Ref, Rc, Arc, Mut, Box))]
pub trait Optimizer {
    /// A point in the problem space being optimized.
    type Point;

    /// Return the best point discovered.
    fn best_point(&self) -> Self::Point;
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_obj_safe;

    use super::*;

    assert_obj_safe!(Optimizer<Point = ()>);
}
