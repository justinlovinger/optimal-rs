use core::{fmt, ops};
use std::marker::PhantomData;

use crate::{
    impl_core_ops,
    math::Mul,
    peano::{One, Two, Zero},
    sum::Sum,
    Computation, ComputationFn, NamedArgs,
};

/// See [`Computation::identity_matrix`].
#[derive(Clone, Copy, Debug)]
pub struct IdentityMatrix<Len, T>
where
    Self: Computation,
{
    pub len: Len,
    pub(super) ty: PhantomData<T>,
}

impl<Len, T> IdentityMatrix<Len, T>
where
    Self: Computation,
{
    pub fn new(len: Len) -> Self {
        Self {
            len,
            ty: PhantomData,
        }
    }
}

impl<Len, T> Computation for IdentityMatrix<Len, T>
where
    Len: Computation<Dim = Zero, Item = usize>,
{
    type Dim = Two;
    type Item = T;
}

impl<Len, T> ComputationFn for IdentityMatrix<Len, T>
where
    Self: Computation,
    Len: ComputationFn,
    IdentityMatrix<Len::Filled, T>: Computation,
{
    type Filled = IdentityMatrix<Len::Filled, T>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        IdentityMatrix {
            len: self.len.fill(named_args),
            ty: self.ty,
        }
    }

    fn arg_names(&self) -> crate::Names {
        self.len.arg_names()
    }
}

impl_core_ops!(IdentityMatrix<Len, T>);

impl<Len, T> fmt::Display for IdentityMatrix<Len, T>
where
    Self: Computation,
    Len: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "identity_matrix({})", self.len)
    }
}

/// A computation representing a diagonal matrix
/// from an element.
#[derive(Clone, Copy, Debug)]
pub struct FromDiagElem<Len, Elem>
where
    Self: Computation,
{
    pub len: Len,
    pub elem: Elem,
}

impl<Len, Elem> FromDiagElem<Len, Elem>
where
    Self: Computation,
{
    #[allow(missing_docs)]
    pub fn new(len: Len, elem: Elem) -> Self {
        Self { len, elem }
    }
}

impl<Len, Elem> Computation for FromDiagElem<Len, Elem>
where
    Len: Computation<Dim = Zero, Item = usize>,
    Elem: Computation<Dim = Zero>,
{
    type Dim = Two;
    type Item = Elem::Item;
}

impl<Len, Elem> ComputationFn for FromDiagElem<Len, Elem>
where
    Self: Computation,
    Len: ComputationFn,
    Elem: ComputationFn,
    FromDiagElem<Len::Filled, Elem::Filled>: Computation,
{
    type Filled = FromDiagElem<Len::Filled, Elem::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1) = named_args
            .partition(&self.len.arg_names(), &self.elem.arg_names())
            .unwrap_or_else(|e| panic!("{}", e,));
        FromDiagElem {
            len: self.len.fill(args_0),
            elem: self.elem.fill(args_1),
        }
    }

    fn arg_names(&self) -> crate::Names {
        self.len.arg_names().union(self.elem.arg_names())
    }
}

impl_core_ops!(FromDiagElem<Len, Elem>);

impl<Len, Elem> fmt::Display for FromDiagElem<Len, Elem>
where
    Self: Computation,
    Len: fmt::Display,
    Elem: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "from_diag_elem({}, {})", self.len, self.elem)
    }
}

/// See [`Computation::scalar_product`].
pub type ScalarProduct<A, B> = Sum<Mul<A, B>>;

/// See [`Computation::scalar_product`].
pub fn scalar_product<A, B>(x: A, y: B) -> ScalarProduct<A, B>
where
    Mul<A, B>: Computation,
    ScalarProduct<A, B>: Computation,
{
    Sum(Mul(x, y))
}

// With better support for overlapping trait-implementations
// we could add the following:
//
// ```
// impl<A, B> fmt::Display for ScalarProduct<A, B>
// where
//     Self: Computation,
//     A: fmt::Display,
//     B: fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "({} . {})", self.0, self.1)
//     }
// }
// ```

/// See [`Computation::mat_mul`].
#[derive(Clone, Copy, Debug)]
pub struct MatMul<A, B>(pub A, pub B)
where
    Self: Computation;

impl<A, B> Computation for MatMul<A, B>
where
    A: Computation<Dim = Two>,
    B: Computation<Dim = Two>,
    A::Item: ops::Mul<B::Item>,
    <A::Item as ops::Mul<B::Item>>::Output: ops::Add,
{
    type Dim = Two;
    type Item = <<A::Item as ops::Mul<B::Item>>::Output as ops::Add>::Output;
}

impl<A, B> ComputationFn for MatMul<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
    MatMul<A::Filled, B::Filled>: Computation,
{
    type Filled = MatMul<A::Filled, B::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1) = named_args
            .partition(&self.0.arg_names(), &self.1.arg_names())
            .unwrap_or_else(|e| panic!("{}", e,));
        MatMul(self.0.fill(args_0), self.1.fill(args_1))
    }

    fn arg_names(&self) -> crate::Names {
        self.0.arg_names().union(self.1.arg_names())
    }
}

impl_core_ops!(MatMul<A, B>);

impl<A, B> fmt::Display for MatMul<A, B>
where
    Self: Computation,
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} x {})", self.0, self.1)
    }
}

/// See [`Computation::mul_out`].
#[derive(Clone, Copy, Debug)]
pub struct MulOut<A, B>(pub A, pub B)
where
    Self: Computation;

impl<A, B> Computation for MulOut<A, B>
where
    A: Computation<Dim = One>,
    B: Computation<Dim = One>,
    A::Item: ops::Mul<B::Item>,
{
    type Dim = Two;
    type Item = <A::Item as ops::Mul<B::Item>>::Output;
}

impl<A, B> ComputationFn for MulOut<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
    MulOut<A::Filled, B::Filled>: Computation,
{
    type Filled = MulOut<A::Filled, B::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1) = named_args
            .partition(&self.0.arg_names(), &self.1.arg_names())
            .unwrap_or_else(|e| panic!("{}", e,));
        MulOut(self.0.fill(args_0), self.1.fill(args_1))
    }

    fn arg_names(&self) -> crate::Names {
        self.0.arg_names().union(self.1.arg_names())
    }
}

impl_core_ops!(MulOut<A, B>);

impl<A, B> fmt::Display for MulOut<A, B>
where
    Self: Computation,
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(col({}) x row({}))", self.0, self.1)
    }
}

/// See [`Computation::mul_col`].
#[derive(Clone, Copy, Debug)]
pub struct MulCol<A, B>(pub A, pub B)
where
    Self: Computation;

impl<A, B> Computation for MulCol<A, B>
where
    A: Computation<Dim = Two>,
    B: Computation<Dim = One>,
    A::Item: ops::Mul<B::Item>,
    <A::Item as ops::Mul<B::Item>>::Output: ops::Add,
{
    type Dim = One;
    type Item = <<A::Item as ops::Mul<B::Item>>::Output as ops::Add>::Output;
}

impl<A, B> ComputationFn for MulCol<A, B>
where
    Self: Computation,
    A: ComputationFn,
    B: ComputationFn,
    MulCol<A::Filled, B::Filled>: Computation,
{
    type Filled = MulCol<A::Filled, B::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        let (args_0, args_1) = named_args
            .partition(&self.0.arg_names(), &self.1.arg_names())
            .unwrap_or_else(|e| panic!("{}", e,));
        MulCol(self.0.fill(args_0), self.1.fill(args_1))
    }

    fn arg_names(&self) -> crate::Names {
        self.0.arg_names().union(self.1.arg_names())
    }
}

impl_core_ops!(MulCol<A, B>);

impl<A, B> fmt::Display for MulCol<A, B>
where
    Self: Computation,
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({} x col({}))", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{linalg::FromDiagElem, run::Matrix, val, val1, val2, Computation};

    #[proptest]
    fn identity_matrix_should_display(x: usize) {
        let inp = val!(x);
        prop_assert_eq!(
            inp.identity_matrix::<i32>().to_string(),
            format!("identity_matrix({})", inp)
        );
    }

    #[proptest]
    fn from_diag_elem_should_display(len: usize, elem: i32) {
        let len = val!(len);
        let elem = val!(elem);
        prop_assert_eq!(
            FromDiagElem::new(len, elem).to_string(),
            format!("from_diag_elem({}, {})", len, elem)
        );
    }

    // With better support for overlapping trait-implementations
    // we could add the following:
    //
    // ```
    // #[proptest]
    // fn scalar_product_should_display(x: i32, y: i32, z: i32, q: i32) {
    //     let lhs = val1!([x, y]);
    //     let rhs = val1!([z, q]);
    //     prop_assert_eq!(
    //         lhs.scalar_product(rhs).to_string(),
    //         format!("({} . {})", lhs, rhs)
    //     );
    // }
    // ```

    #[proptest]
    fn mat_mul_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val2!(Matrix::from_vec((2, 1), vec![x, y]).unwrap());
        let rhs = val2!(Matrix::from_vec((1, 2), vec![z, q]).unwrap());
        prop_assert_eq!(
            lhs.clone().mat_mul(rhs.clone()).to_string(),
            format!("({} x {})", lhs, rhs)
        );
    }

    #[proptest]
    fn mul_out_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val1!([x, y]);
        let rhs = val1!([z, q]);
        prop_assert_eq!(
            lhs.mul_out(rhs).to_string(),
            format!("(col({}) x row({}))", lhs, rhs)
        );
    }

    #[proptest]
    fn mul_col_should_display(x: i32, y: i32, z: i32, q: i32) {
        let lhs = val2!(Matrix::from_vec((2, 1), vec![x, y]).unwrap());
        let rhs = val1!([z, q]);
        prop_assert_eq!(
            lhs.clone().mul_col(rhs).to_string(),
            format!("({} x col({}))", lhs, rhs)
        );
    }
}
