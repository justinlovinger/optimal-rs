pub trait IntoVec {
    type Item;

    fn into_vec(self) -> Vec<Self::Item>;
}

impl<A> IntoVec for Vec<A> {
    type Item = A;

    fn into_vec(self) -> Vec<Self::Item> {
        self
    }
}

impl<A, const LEN: usize> IntoVec for [A; LEN] {
    type Item = A;

    fn into_vec(self) -> Vec<Self::Item> {
        self.into()
    }
}

impl<I, F> IntoVec for std::iter::Map<I, F>
where
    Self: Iterator,
{
    type Item = <Self as Iterator>::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}

impl<I> IntoVec for std::iter::Cloned<I>
where
    Self: Iterator,
{
    type Item = <Self as Iterator>::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}

impl<I> IntoVec for std::iter::Copied<I>
where
    Self: Iterator,
{
    type Item = <Self as Iterator>::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}

impl<I> IntoVec for std::iter::Take<I>
where
    Self: Iterator,
{
    type Item = <Self as Iterator>::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}

impl<T> IntoVec for std::ops::Range<T>
where
    Self: Iterator,
{
    type Item = <Self as Iterator>::Item;

    fn into_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}
