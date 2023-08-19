/// A type with a default value
/// parameterized on another value.
pub trait DefaultFor<T> {
    /// Return a default value for `x`.
    fn default_for(x: T) -> Self;
}

impl<A, T> DefaultFor<A> for T
where
    T: Default,
{
    fn default_for(_: A) -> Self {
        T::default()
    }
}
