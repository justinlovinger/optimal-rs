/// A type containing the value of a point.
pub trait Value<A> {
    /// Return owned value.
    fn into_value(self) -> A;

    /// Return value.
    fn value(&self) -> &A;
}

impl<A> Value<A> for A {
    fn into_value(self) -> A {
        self
    }

    fn value(&self) -> &A {
        self
    }
}

impl<A> Value<A> for (A, Vec<A>) {
    fn into_value(self) -> A {
        self.0
    }

    fn value(&self) -> &A {
        &self.0
    }
}

/// A type containing the derivatives of a point.
pub trait Derivatives<A> {
    /// Return owned derivatives.
    fn into_derivatives(self) -> Vec<A>;

    /// Return derivatives.
    fn derivatives(&self) -> &[A];
}

impl<A> Derivatives<A> for Vec<A> {
    fn into_derivatives(self) -> Vec<A> {
        self
    }

    fn derivatives(&self) -> &[A] {
        self
    }
}

impl<A> Derivatives<A> for (A, Vec<A>) {
    fn into_derivatives(self) -> Vec<A> {
        self.1
    }

    fn derivatives(&self) -> &[A] {
        &self.1
    }
}

/// A type containing the value and derivatives of a point.
pub trait ValueDerivatives<A>: Value<A> + Derivatives<A> {
    /// Return owned value and derivatives.
    fn into_value_derivatives(self) -> (A, Vec<A>);
}

impl<A> ValueDerivatives<A> for (A, Vec<A>) {
    fn into_value_derivatives(self) -> (A, Vec<A>) {
        self
    }
}
