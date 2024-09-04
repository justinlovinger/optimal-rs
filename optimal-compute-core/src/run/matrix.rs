#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matrix<V> {
    shape: (usize, usize),
    inner: V,
}

impl<A> Matrix<Vec<A>> {
    pub fn from_vec(shape: (usize, usize), vec: Vec<A>) -> Option<Self> {
        if shape.0 * shape.1 == vec.len() {
            Some(Matrix { shape, inner: vec })
        } else {
            None
        }
    }
}

impl<I> Matrix<I> {
    pub fn from_iter(shape: (usize, usize), it: I) -> Option<Self>
    where
        I: ExactSizeIterator,
    {
        if shape.0 * shape.1 == it.len() {
            Some(Matrix { shape, inner: it })
        } else {
            None
        }
    }
}

impl<V> Matrix<V> {
    /// # Safety
    ///
    /// This function is safe
    /// if `inner` implements `IntoIterator`
    /// and produces a number of items
    /// equal to `shape.0 * shape.1`.
    pub unsafe fn new_unchecked(shape: (usize, usize), inner: V) -> Self {
        Matrix { shape, inner }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn inner(&self) -> &V {
        &self.inner
    }

    pub fn into_inner(self) -> V {
        self.inner
    }
}

impl<V> IntoIterator for Matrix<V>
where
    V: IntoIterator,
{
    type Item = V::Item;

    type IntoIter = V::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}
