#![feature(min_specialization)]
#![feature(unsized_fn_params)]
#![allow(internal_features)]
#![warn(missing_debug_implementations)]

//! Types for abstract mathematical computation.
//!
//! Note,
//! this framework is highly experimental.
//! Documentation is lacking
//! and breaking changes are expected.
//! The best way to learn about this framework
//! is to read the tests
//! and see how it is used to implement algorithms
//! in algorithm-specific packages.
//!
//! # Examples
//!
//! ```
//! use computation_types::{named_args, val, Run};
//!
//! let one_plus_one = val!(1) + val!(1);
//! assert_eq!(one_plus_one.to_string(), "(1 + 1)");
//! assert_eq!(one_plus_one.run(), 2);
//! ```

mod function;
pub mod macros;
mod named_args;
mod names;
pub mod peano;
pub mod run;

pub mod black_box;
pub mod cmp;
pub mod control_flow;
pub mod enumerate;
pub mod linalg;
pub mod math;
pub mod rand;
pub mod sum;
pub mod zip;

use core::fmt;
use std::marker::PhantomData;

use crate::peano::{One, Suc, Two, Zero};

pub use crate::{function::*, named_args::*, names::*, run::Run};

/// A type representing a computation.
///
/// This trait does little on its own.
/// Additional traits,
/// such as [`Run`],
/// must be implemented
/// to use a computation.
#[allow(clippy::len_without_is_empty)]
pub trait Computation {
    type Dim;
    type Item;

    // `math`

    fn add<Rhs>(self, rhs: Rhs) -> math::Add<Self, Rhs>
    where
        Self: Sized,
        math::Add<Self, Rhs>: Computation,
    {
        math::Add(self, rhs)
    }

    fn sub<Rhs>(self, rhs: Rhs) -> math::Sub<Self, Rhs>
    where
        Self: Sized,
        math::Sub<Self, Rhs>: Computation,
    {
        math::Sub(self, rhs)
    }

    fn mul<Rhs>(self, rhs: Rhs) -> math::Mul<Self, Rhs>
    where
        Self: Sized,
        math::Mul<Self, Rhs>: Computation,
    {
        math::Mul(self, rhs)
    }

    fn div<Rhs>(self, rhs: Rhs) -> math::Div<Self, Rhs>
    where
        Self: Sized,
        math::Div<Self, Rhs>: Computation,
    {
        math::Div(self, rhs)
    }

    fn pow<Rhs>(self, rhs: Rhs) -> math::Pow<Self, Rhs>
    where
        Self: Sized,
        math::Pow<Self, Rhs>: Computation,
    {
        math::Pow(self, rhs)
    }

    fn neg(self) -> math::Neg<Self>
    where
        Self: Sized,
        math::Neg<Self>: Computation,
    {
        math::Neg(self)
    }

    fn abs(self) -> math::Abs<Self>
    where
        Self: Sized,
        math::Abs<Self>: Computation,
    {
        math::Abs(self)
    }

    // `math::trig`

    fn sin(self) -> math::Sin<Self>
    where
        Self: Sized,
        math::Sin<Self>: Computation,
    {
        math::Sin(self)
    }

    fn cos(self) -> math::Cos<Self>
    where
        Self: Sized,
        math::Cos<Self>: Computation,
    {
        math::Cos(self)
    }

    fn tan(self) -> math::Tan<Self>
    where
        Self: Sized,
        math::Tan<Self>: Computation,
    {
        math::Tan(self)
    }

    fn asin(self) -> math::Asin<Self>
    where
        Self: Sized,
        math::Asin<Self>: Computation,
    {
        math::Asin(self)
    }

    fn acos(self) -> math::Acos<Self>
    where
        Self: Sized,
        math::Acos<Self>: Computation,
    {
        math::Acos(self)
    }

    fn atan(self) -> math::Atan<Self>
    where
        Self: Sized,
        math::Atan<Self>: Computation,
    {
        math::Atan(self)
    }

    // `cmp`

    fn eq<Rhs>(self, rhs: Rhs) -> cmp::Eq<Self, Rhs>
    where
        Self: Sized,
        cmp::Eq<Self, Rhs>: Computation,
    {
        cmp::Eq(self, rhs)
    }

    fn ne<Rhs>(self, rhs: Rhs) -> cmp::Ne<Self, Rhs>
    where
        Self: Sized,
        cmp::Ne<Self, Rhs>: Computation,
    {
        cmp::Ne(self, rhs)
    }

    fn lt<Rhs>(self, rhs: Rhs) -> cmp::Lt<Self, Rhs>
    where
        Self: Sized,
        cmp::Lt<Self, Rhs>: Computation,
    {
        cmp::Lt(self, rhs)
    }

    fn le<Rhs>(self, rhs: Rhs) -> cmp::Le<Self, Rhs>
    where
        Self: Sized,
        cmp::Le<Self, Rhs>: Computation,
    {
        cmp::Le(self, rhs)
    }

    fn gt<Rhs>(self, rhs: Rhs) -> cmp::Gt<Self, Rhs>
    where
        Self: Sized,
        cmp::Gt<Self, Rhs>: Computation,
    {
        cmp::Gt(self, rhs)
    }

    fn ge<Rhs>(self, rhs: Rhs) -> cmp::Ge<Self, Rhs>
    where
        Self: Sized,
        cmp::Ge<Self, Rhs>: Computation,
    {
        cmp::Ge(self, rhs)
    }

    fn max(self) -> cmp::Max<Self>
    where
        Self: Sized,
        cmp::Max<Self>: Computation,
    {
        cmp::Max(self)
    }

    fn not(self) -> cmp::Not<Self>
    where
        Self: Sized,
        cmp::Not<Self>: Computation,
    {
        cmp::Not(self)
    }

    // `enumerate`

    fn enumerate<F>(self, f: Function<(Name, Name), F>) -> enumerate::Enumerate<Self, F>
    where
        Self: Sized,
        enumerate::Enumerate<Self, F>: Computation,
    {
        enumerate::Enumerate { child: self, f }
    }

    // `sum`

    fn sum(self) -> sum::Sum<Self>
    where
        Self: Sized,
        sum::Sum<Self>: Computation,
    {
        sum::Sum(self)
    }

    // `zip`

    fn zip<Rhs>(self, rhs: Rhs) -> zip::Zip<Self, Rhs>
    where
        Self: Sized,
        zip::Zip<Self, Rhs>: Computation,
    {
        zip::Zip(self, rhs)
    }

    fn fst(self) -> zip::Fst<Self>
    where
        Self: Sized,
        zip::Fst<Self>: Computation,
    {
        zip::Fst(self)
    }

    fn snd(self) -> zip::Snd<Self>
    where
        Self: Sized,
        zip::Snd<Self>: Computation,
    {
        zip::Snd(self)
    }

    // `black_box`

    /// Run the given regular function `F`.
    ///
    /// This acts as an escape-hatch to allow regular Rust-code in a computation,
    /// but the computation may lose features or efficiency if it is used.
    fn black_box<F, FDim, FItem>(self, f: F) -> black_box::BlackBox<Self, F, FDim, FItem>
    where
        Self: Sized,
        black_box::BlackBox<Self, F, FDim, FItem>: Computation,
    {
        black_box::BlackBox {
            child: self,
            f,
            f_dim: PhantomData,
            f_item: PhantomData,
        }
    }

    // `control_flow`

    fn if_<ArgNames, P, FTrue, FFalse>(
        self,
        arg_names: ArgNames,
        predicate: P,
        f_true: FTrue,
        f_false: FFalse,
    ) -> control_flow::If<Self, ArgNames, P, FTrue, FFalse>
    where
        Self: Sized,
        control_flow::If<Self, ArgNames, P, FTrue, FFalse>: Computation,
    {
        control_flow::If {
            child: self,
            arg_names,
            predicate,
            f_true,
            f_false,
        }
    }

    fn loop_while<ArgNames, F, P>(
        self,
        arg_names: ArgNames,
        f: F,
        predicate: P,
    ) -> control_flow::LoopWhile<Self, ArgNames, F, P>
    where
        Self: Sized,
        control_flow::LoopWhile<Self, ArgNames, F, P>: Computation,
    {
        control_flow::LoopWhile {
            child: self,
            arg_names,
            f,
            predicate,
        }
    }

    fn then<ArgNames, F>(
        self,
        f: function::Function<ArgNames, F>,
    ) -> control_flow::Then<Self, ArgNames, F>
    where
        Self: Sized,
        control_flow::Then<Self, ArgNames, F>: Computation,
    {
        control_flow::Then { child: self, f }
    }

    // `linalg`

    /// Return a `self` by `self` identity-matrix.
    ///
    /// Diagonal elements have a value of `1`,
    /// and non-diagonal elements have a value of `0`.
    ///
    /// The type of elements,
    /// `T`,
    /// may need to be specified.
    fn identity_matrix<T>(self) -> linalg::IdentityMatrix<Self, T>
    where
        Self: Sized,
        linalg::IdentityMatrix<Self, T>: Computation,
    {
        linalg::IdentityMatrix {
            len: self,
            ty: PhantomData::<T>,
        }
    }

    /// Multiply and sum the elements of two vectors.
    ///
    /// This is sometimes known as the "inner product"
    /// or "dot product".
    fn scalar_product<Rhs>(self, rhs: Rhs) -> linalg::ScalarProduct<Self, Rhs>
    where
        Self: Sized,
        linalg::ScalarProduct<Self, Rhs>: Computation,
    {
        linalg::ScalarProduct(self, rhs)
    }

    /// Perform matrix-multiplication.
    fn mat_mul<Rhs>(self, rhs: Rhs) -> linalg::MatMul<Self, Rhs>
    where
        Self: Sized,
        linalg::MatMul<Self, Rhs>: Computation,
    {
        linalg::MatMul(self, rhs)
    }

    /// Multiply elements from the Cartesian product of two vectors.
    ///
    /// This is sometimes known as "outer product",
    /// and it is equivalent to matrix-multiplying a column-matrix by a row-matrix.
    fn mul_out<Rhs>(self, rhs: Rhs) -> linalg::MulOut<Self, Rhs>
    where
        Self: Sized,
        linalg::MulOut<Self, Rhs>: Computation,
    {
        linalg::MulOut(self, rhs)
    }

    /// Matrix-multiply a matrix by a column-matrix,
    /// returning a vector.
    fn mul_col<Rhs>(self, rhs: Rhs) -> linalg::MulCol<Self, Rhs>
    where
        Self: Sized,
        linalg::MulCol<Self, Rhs>: Computation,
    {
        linalg::MulCol(self, rhs)
    }

    // Other

    fn len(self) -> Len<Self>
    where
        Self: Sized,
        Len<Self>: Computation,
    {
        Len(self)
    }
}

impl<T> Computation for &T
where
    T: Computation + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

impl<T> Computation for &mut T
where
    T: Computation + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

impl<T> Computation for Box<T>
where
    T: Computation + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

impl<T> Computation for std::rc::Rc<T>
where
    T: Computation + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

impl<T> Computation for std::sync::Arc<T>
where
    T: Computation + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

impl<T> Computation for std::borrow::Cow<'_, T>
where
    T: Computation + ToOwned + ?Sized,
{
    type Dim = T::Dim;
    type Item = T::Item;
}

/// A type representing a function-like computation.
///
/// Most computations should implement this,
/// even if they represent a function with zero arguments.
pub trait ComputationFn: Computation {
    type Filled;

    /// Fill arguments will values,
    /// replacing `Arg`s with `Val`s.
    fn fill(self, named_args: NamedArgs) -> Self::Filled;

    fn arg_names(&self) -> Names;
}

impl<T> ComputationFn for &T
where
    T: ComputationFn + ToOwned + ?Sized,
    T::Owned: ComputationFn,
{
    type Filled = <T::Owned as ComputationFn>::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        self.to_owned().fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

impl<T> ComputationFn for &mut T
where
    T: ComputationFn + ToOwned + ?Sized,
    T::Owned: ComputationFn,
{
    type Filled = <T::Owned as ComputationFn>::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        self.to_owned().fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

impl<T> ComputationFn for Box<T>
where
    T: ComputationFn + ?Sized,
{
    type Filled = T::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        (*self).fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

impl<T> ComputationFn for std::rc::Rc<T>
where
    T: ComputationFn + ToOwned + ?Sized,
    T::Owned: ComputationFn,
{
    type Filled = <T::Owned as ComputationFn>::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        self.as_ref().to_owned().fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

impl<T> ComputationFn for std::sync::Arc<T>
where
    T: ComputationFn + ToOwned + ?Sized,
    T::Owned: ComputationFn,
{
    type Filled = <T::Owned as ComputationFn>::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        self.as_ref().to_owned().fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

impl<T> ComputationFn for std::borrow::Cow<'_, T>
where
    T: ComputationFn + ToOwned + ?Sized,
    T::Owned: ComputationFn,
{
    type Filled = <T::Owned as ComputationFn>::Filled;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        self.into_owned().fill(named_args)
    }

    fn arg_names(&self) -> Names {
        (*(*self)).arg_names()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Val<Dim, A>
where
    Self: Computation,
{
    dim: PhantomData<Dim>,
    pub inner: A,
}

#[derive(Clone, Copy, Debug)]
pub struct Arg<Dim, A>
where
    Self: Computation,
{
    pub name: &'static str,
    dim: PhantomData<Dim>,
    elem: PhantomData<A>,
}

pub type Val0<A> = Val<Zero, A>;
pub type Val1<A> = Val<One, A>;
pub type Val2<A> = Val<Two, A>;
pub type Arg0<A> = Arg<Zero, A>;
pub type Arg1<A> = Arg<One, A>;
pub type Arg2<A> = Arg<Two, A>;

impl<Dim, A> Val<Dim, A>
where
    Self: Computation,
{
    pub fn new(value: A) -> Self {
        Val {
            dim: PhantomData,
            inner: value,
        }
    }
}

impl<Dim, A> Arg<Dim, A> {
    pub fn new(name: &'static str) -> Self {
        Arg {
            name,
            dim: PhantomData,
            elem: PhantomData,
        }
    }
}

#[macro_export]
macro_rules! val {
    ( $value:expr ) => {
        $crate::Val0::new($value)
    };
}

#[macro_export]
macro_rules! val1 {
    ( $value:expr ) => {
        $crate::Val1::new($value)
    };
}

#[macro_export]
macro_rules! val2 {
    ( $value:expr ) => {
        $crate::Val2::new($value)
    };
}

#[macro_export]
macro_rules! arg {
    ( $name:literal ) => {
        $crate::Arg0::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg0::<$elem>::new($name)
    };
}

#[macro_export]
macro_rules! arg1 {
    ( $name:literal ) => {
        $crate::Arg1::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg1::<$elem>::new($name)
    };
}

#[macro_export]
macro_rules! arg2 {
    ( $name:literal ) => {
        $crate::Arg2::new($name)
    };
    ( $name:literal, $elem:ty ) => {
        $crate::Arg2::<$elem>::new($name)
    };
}

impl<A> Computation for Val<Zero, A> {
    type Dim = Zero;
    type Item = A;
}

impl<D, A> Computation for Val<Suc<D>, A>
where
    A: IntoIterator,
{
    type Dim = Suc<D>;
    type Item = A::Item;
}

impl<D, A> ComputationFn for Val<D, A>
where
    Val<D, A>: Computation,
{
    type Filled = Self;

    fn fill(self, _named_args: NamedArgs) -> Self::Filled {
        self
    }

    fn arg_names(&self) -> Names {
        Names::new()
    }
}

impl<D, A> Computation for Arg<D, A> {
    type Dim = D;
    type Item = A;
}

impl<A> ComputationFn for Arg<Zero, A>
where
    Self: Computation,
    A: 'static + AnyArg,
{
    type Filled = Val<Zero, A>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val {
            dim: self.dim,
            inner: named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        }
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl<A> ComputationFn for Arg<One, A>
where
    Self: Computation,
    A: 'static + Clone + AnyArg,
{
    type Filled = Val<One, Vec<A>>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val {
            dim: self.dim,
            inner: named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        }
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl<A> ComputationFn for Arg<Two, A>
where
    Self: Computation,
    A: 'static + Clone + AnyArg,
{
    type Filled = Val<Two, crate::run::Matrix<Vec<A>>>;

    fn fill(self, mut named_args: NamedArgs) -> Self::Filled {
        Val {
            dim: self.dim,
            inner: named_args
                .pop(self.name)
                .unwrap_or_else(|e| panic!("{}", e)),
        }
    }

    fn arg_names(&self) -> Names {
        Names::singleton(self.name)
    }
}

impl_core_ops!(Val<Dim, A>);
impl_core_ops!(Arg<Dim, A>);

impl<A> fmt::Display for Val<Zero, A>
where
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<D, A> fmt::Display for Val<Suc<D>, A>
where
    Self: Computation,
    A: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl<Dim, A> fmt::Display for Arg<Dim, A>
where
    Self: Computation,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Len<A>(pub A)
where
    Self: Computation;

impl<A> Computation for Len<A>
where
    A: Computation<Dim = One>,
{
    type Dim = Zero;
    type Item = usize;
}

impl<A> ComputationFn for Len<A>
where
    Self: Computation,
    A: ComputationFn,
    Len<A::Filled>: Computation,
{
    type Filled = Len<A::Filled>;

    fn fill(self, named_args: NamedArgs) -> Self::Filled {
        Len(self.0.fill(named_args))
    }

    fn arg_names(&self) -> Names {
        self.0.arg_names()
    }
}

impl_core_ops!(Len<A>);

impl<A> fmt::Display for Len<A>
where
    Self: Computation,
    A: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.len()", self.0)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn val_should_display_inner(x: i32) {
        prop_assert_eq!(val!(x).to_string(), x.to_string())
    }

    #[proptest]
    fn val1_should_display_items(xs: Vec<i32>) {
        prop_assert_eq!(val1!(xs.clone()).to_string(), format!("{:?}", xs.clone()));
    }

    #[test]
    fn arg_should_display_placeholder() {
        assert_eq!(arg!("foo", i32).to_string(), "foo");
        assert_eq!(arg1!("bar", i32).to_string(), "bar");
    }

    #[test]
    fn len_should_display() {
        let inp = val1!(vec![0, 1]);
        assert_eq!(inp.clone().len().to_string(), format!("{}.len()", inp));
    }

    // The following test requires `Eq` for computation-types:
    // ```
    // #[proptest]
    // fn args_should_propagate_correctly(
    //     #[strategy(-1000..1000)] x: i32,
    //     #[strategy(-1000..1000)] y: i32,
    //     #[strategy(-1000..1000)] z: i32,
    //     #[strategy(-1000..1000)] in_x: i32,
    //     #[strategy(-1000..1000)] in_y: i32,
    //     #[strategy(-1000..1000)] in_z: i32,
    // ) {
    //     prop_assume!((x - in_y) != 0);
    //     prop_assume!(z != 0);
    //     prop_assert_eq!(
    //         (arg!("foo", i32) / (val!(x) - arg!("bar", i32))
    //             + -(val!(z) * val!(y) + arg!("baz", i32)))
    //         .fill(named_args![("foo", in_x), ("bar", in_y), ("baz", in_z)]),
    //         val!(in_x) / (val!(x) - val!(in_y)) + -(val!(z) * val!(y) + val!(in_z))
    //     );
    //     prop_assert_eq!(
    //         (arg!("foo", i32)
    //             + (((val!(x) + val!(y) - arg!("bar", i32)) / -val!(z)) * arg!("baz", i32)))
    //         .fill(named_args![("foo", in_x), ("bar", in_y), ("baz", in_z)]),
    //         val!(in_x) + (((val!(x) + val!(y) - val!(in_y)) / -val!(z)) * val!(in_z))
    //     );
    //     prop_assert_eq!(
    //         -(-arg!("foo", i32)).fill(named_args![("foo", x)]),
    //         -(-val!(x))
    //     );
    // }
    // ```

    mod dynamic {
        use ::rand::distributions::Uniform;
        use run::RunCore;
        use zip::{Zip, Zip3, Zip4};

        use self::rand::Rand;

        use super::*;

        #[test]
        fn the_framework_should_support_dynamic_objective_functions() {
            trait ObjFunc:
                ComputationFn<Dim = Zero, Item = f64, Filled = Box<dyn FilledObjFunc>>
            {
                fn boxed_clone(&self) -> Box<dyn ObjFunc>;
            }
            impl<T> ObjFunc for T
            where
                T: 'static
                    + Clone
                    + ComputationFn<Dim = Zero, Item = f64, Filled = Box<dyn FilledObjFunc>>,
            {
                fn boxed_clone(&self) -> Box<dyn ObjFunc> {
                    Box::new(self.clone())
                }
            }
            impl Clone for Box<dyn ObjFunc> {
                fn clone(&self) -> Self {
                    self.as_ref().boxed_clone()
                }
            }

            trait FilledObjFunc: Computation<Dim = Zero, Item = f64> + RunCore<Output = f64> {
                fn boxed_clone(&self) -> Box<dyn FilledObjFunc>;
            }
            impl<T> FilledObjFunc for T
            where
                T: 'static + Clone + Computation<Dim = Zero, Item = f64> + RunCore<Output = f64>,
            {
                fn boxed_clone(&self) -> Box<dyn FilledObjFunc> {
                    Box::new(self.clone())
                }
            }
            impl Clone for Box<dyn FilledObjFunc> {
                fn clone(&self) -> Self {
                    self.as_ref().boxed_clone()
                }
            }

            fn random_optimizer(
                len: usize,
                samples: usize,
                obj_func: Box<dyn ObjFunc>,
            ) -> impl Run<Output = Vec<f64>> {
                let distr = std::iter::repeat(Uniform::new(0.0, 1.0))
                    .take(len)
                    .collect::<Vec<_>>();
                Zip(
                    val!(1_usize),
                    Rand::<Val1<Vec<Uniform<f64>>>, f64>::new(val1!(distr.clone())).then(
                        Function::anonymous("point", Zip(arg1!("point", f64), obj_func.clone())),
                    ),
                )
                .loop_while(
                    ("i", ("best_point", "best_value")),
                    Zip(
                        arg!("i", usize) + val!(1_usize),
                        Zip3(
                            arg1!("best_point", f64),
                            arg!("best_value", f64),
                            Rand::<Val1<Vec<Uniform<f64>>>, f64>::new(val1!(distr)),
                        )
                        .then(Function::anonymous(
                            ("best_point", "best_value", "point"),
                            Zip4(
                                arg1!("best_point", f64),
                                arg!("best_value", f64),
                                arg1!("point", f64),
                                obj_func,
                            )
                            .then(Function::anonymous(
                                ("best_point", "best_value", "point", "value"),
                                Zip4(
                                    arg1!("best_point", f64),
                                    arg!("best_value", f64),
                                    arg1!("point", f64),
                                    arg!("value", f64),
                                )
                                .if_(
                                    ("best_point", "best_value", "point", "value"),
                                    arg!("value", f64).lt(arg!("best_value", f64)),
                                    Zip(arg1!("point", f64), arg!("value", f64)),
                                    Zip(arg1!("best_point", f64), arg!("best_value", f64)),
                                ),
                            )),
                        )),
                    ),
                    arg!("i", usize).lt(val!(samples)),
                )
                .then(Function::anonymous(
                    ("i", ("best_point", "best_value")),
                    arg1!("best_point", f64),
                ))
            }

            #[derive(Clone, Copy, Debug)]
            struct BoxFillObjFunc<A>(A);

            impl<A> Computation for BoxFillObjFunc<A>
            where
                A: Computation,
            {
                type Dim = A::Dim;
                type Item = A::Item;
            }

            impl<A> ComputationFn for BoxFillObjFunc<A>
            where
                A: ComputationFn,
                A::Filled: 'static + FilledObjFunc,
            {
                type Filled = Box<dyn FilledObjFunc>;

                fn fill(self, named_args: NamedArgs) -> Self::Filled {
                    Box::new(self.0.fill(named_args))
                }

                fn arg_names(&self) -> Names {
                    self.0.arg_names()
                }
            }

            random_optimizer(2, 10, Box::new(BoxFillObjFunc(arg1!("point", f64).sum()))).run();
        }
    }
}
