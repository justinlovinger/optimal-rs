use core::fmt;

use crate::{impl_core_ops, peano::Zero, Computation, ComputationFn};

#[derive(Clone, Copy, Debug)]
pub struct If<A, Args, P, FTrue, FFalse>
where
    Self: Computation,
{
    pub child: A,
    pub args: Args,
    pub predicate: P,
    pub f_true: FTrue,
    pub f_false: FFalse,
}

impl<A, Args, P, FTrue, FFalse, FDim, FItem> Computation for If<A, Args, P, FTrue, FFalse>
where
    A: Computation,
    P: ComputationFn<Dim = Zero, Item = bool>,
    FTrue: ComputationFn<Dim = FDim, Item = FItem>,
    FFalse: ComputationFn<Dim = FDim, Item = FItem>,
{
    type Dim = FDim;
    type Item = FItem;
}

impl<A, Args, P, FTrue, FFalse> ComputationFn for If<A, Args, P, FTrue, FFalse>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(If<A, Args, P, FTrue, FFalse>);

impl<A, Args, P, FTrue, FFalse> fmt::Display for If<A, Args, P, FTrue, FFalse>
where
    Self: Computation,
    A: fmt::Display,
    Args: fmt::Debug,
    P: fmt::Display,
    FTrue: fmt::Display,
    FFalse: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "let {:?} = {}; if {} {{ {} }} else {{ {} }})",
            self.args, self.child, self.predicate, self.f_true, self.f_false
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoopWhile<A, Args, F, P>
where
    Self: Computation,
{
    pub child: A,
    pub args: Args,
    pub f: F,
    pub predicate: P,
}

impl<A, Args, F, P> Computation for LoopWhile<A, Args, F, P>
where
    A: Computation,
    F: ComputationFn<Dim = A::Dim, Item = A::Item>,
    P: ComputationFn<Dim = Zero, Item = bool>,
{
    type Dim = F::Dim;
    type Item = F::Item;
}

impl<A, Args, F, P> ComputationFn for LoopWhile<A, Args, F, P>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(LoopWhile<A, Args, F, P>);

impl<A, Args, F, P> fmt::Display for LoopWhile<A, Args, F, P>
where
    Self: Computation,
    A: fmt::Display,
    Args: fmt::Debug,
    F: fmt::Display,
    P: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "let {:?} = {}; while !{} {{ x = {}; }})",
            self.args, self.child, self.predicate, self.f
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Then<A, Args, F>
where
    Self: Computation,
{
    pub child: A,
    pub args: Args,
    pub f: F,
}

impl<A, Args, F> Computation for Then<A, Args, F>
where
    A: Computation,
    F: ComputationFn,
{
    type Dim = F::Dim;
    type Item = F::Item;
}

impl<A, Args, F> ComputationFn for Then<A, Args, F>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(Then<A, Args, F>);

impl<A, Args, F> fmt::Display for Then<A, Args, F>
where
    Self: Computation,
    A: fmt::Display,
    Args: fmt::Debug,
    F: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "let {:?} = {}; {}", self.args, self.child, self.f)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg, val, Computation};

    #[proptest]
    fn if_should_display(x: i32) {
        let inp = val!(x);
        let args = ("x",);
        let p = arg!("x", i32).ge(val!(0));
        let f_true = arg!("x", i32) + val!(1);
        let f_false = arg!("x", i32) - val!(1);
        prop_assert_eq!(
            inp.if_(args, p, f_true, f_false).to_string(),
            format!(
                "let {:?} = {}; if {} {{ {} }} else {{ {} }})",
                args, inp, p, f_true, f_false
            )
        );
    }

    #[proptest]
    fn loop_while_should_display(x: i32) {
        let inp = val!(x);
        let args = ("x",);
        let f = arg!("x", i32) + val!(1);
        let p = arg!("x", i32).lt(val!(10));
        prop_assert_eq!(
            inp.loop_while(args, f, p).to_string(),
            format!("let {:?} = {}; while !{} {{ x = {}; }})", args, inp, p, f)
        );
    }

    #[proptest]
    fn then_should_display(x: i32) {
        let inp = val!(x);
        let args = ("x",);
        let f = arg!("x", i32) + val!(1);
        prop_assert_eq!(
            inp.then(args, f).to_string(),
            format!("let {:?} = {}; {}", args, inp, f)
        );
    }
}
