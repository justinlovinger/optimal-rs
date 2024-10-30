use core::fmt;

use crate::{function::Function, impl_core_ops, peano::Zero, Computation, ComputationFn};

#[derive(Clone, Copy, Debug)]
pub struct If<A, ArgNames, P, FTrue, FFalse>
where
    Self: Computation,
{
    pub child: A,
    pub arg_names: ArgNames,
    pub predicate: P,
    pub f_true: FTrue,
    pub f_false: FFalse,
}

impl<A, ArgNames, P, FTrue, FFalse, FDim, FItem> Computation for If<A, ArgNames, P, FTrue, FFalse>
where
    A: Computation,
    P: ComputationFn<Dim = Zero, Item = bool>,
    FTrue: ComputationFn<Dim = FDim, Item = FItem>,
    FFalse: ComputationFn<Dim = FDim, Item = FItem>,
{
    type Dim = FDim;
    type Item = FItem;
}

impl<A, ArgNames, P, FTrue, FFalse> ComputationFn for If<A, ArgNames, P, FTrue, FFalse>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(If<A, ArgNames, P, FTrue, FFalse>);

impl<A, ArgNames, P, FTrue, FFalse> fmt::Display for If<A, ArgNames, P, FTrue, FFalse>
where
    Self: Computation,
    A: fmt::Display,
    ArgNames: fmt::Debug,
    P: fmt::Display,
    FTrue: fmt::Display,
    FFalse: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "let {:?} = {}; if {} {{ {} }} else {{ {} }})",
            self.arg_names, self.child, self.predicate, self.f_true, self.f_false
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoopWhile<A, ArgNames, F, P>
where
    Self: Computation,
{
    pub child: A,
    pub arg_names: ArgNames,
    pub f: F,
    pub predicate: P,
}

impl<A, ArgNames, F, P> Computation for LoopWhile<A, ArgNames, F, P>
where
    A: Computation,
    F: ComputationFn<Dim = A::Dim, Item = A::Item>,
    P: ComputationFn<Dim = Zero, Item = bool>,
{
    type Dim = F::Dim;
    type Item = F::Item;
}

impl<A, ArgNames, F, P> ComputationFn for LoopWhile<A, ArgNames, F, P>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(LoopWhile<A, ArgNames, F, P>);

impl<A, ArgNames, F, P> fmt::Display for LoopWhile<A, ArgNames, F, P>
where
    Self: Computation,
    A: fmt::Display,
    ArgNames: fmt::Debug,
    F: fmt::Display,
    P: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "let {:?} = {}; while !{} {{ x = {}; }})",
            self.arg_names, self.child, self.predicate, self.f
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Then<A, ArgNames, F>
where
    Self: Computation,
{
    pub child: A,
    pub f: Function<ArgNames, F>,
}

impl<A, ArgNames, F> Computation for Then<A, ArgNames, F>
where
    A: Computation,
    F: ComputationFn,
{
    type Dim = F::Dim;
    type Item = F::Item;
}

impl<A, ArgNames, F> ComputationFn for Then<A, ArgNames, F>
where
    Self: Computation,
    A: ComputationFn,
{
    fn arg_names(&self) -> crate::Names {
        self.child.arg_names()
    }
}

impl_core_ops!(Then<A, ArgNames, F>);

impl<A, ArgNames, F> fmt::Display for Then<A, ArgNames, F>
where
    Self: Computation,
    A: fmt::Display,
    ArgNames: fmt::Debug,
    F: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "let {:?} = {}; {}",
            self.f.arg_names, self.child, self.f.body
        )
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use test_strategy::proptest;

    use crate::{arg, function::Function, val, Computation};

    #[proptest]
    fn if_should_display(x: i32) {
        let inp = val!(x);
        let arg_names = ("x",);
        let p = arg!("x", i32).ge(val!(0));
        let f_true = arg!("x", i32) + val!(1);
        let f_false = arg!("x", i32) - val!(1);
        prop_assert_eq!(
            inp.if_(arg_names, p, f_true, f_false).to_string(),
            format!(
                "let {:?} = {}; if {} {{ {} }} else {{ {} }})",
                arg_names, inp, p, f_true, f_false
            )
        );
    }

    #[proptest]
    fn loop_while_should_display(x: i32) {
        let inp = val!(x);
        let arg_names = ("x",);
        let f = arg!("x", i32) + val!(1);
        let p = arg!("x", i32).lt(val!(10));
        prop_assert_eq!(
            inp.loop_while(arg_names, f, p).to_string(),
            format!(
                "let {:?} = {}; while !{} {{ x = {}; }})",
                arg_names, inp, p, f
            )
        );
    }

    #[proptest]
    fn then_should_display(x: i32) {
        let inp = val!(x);
        let arg_names = ("x",);
        let f = arg!("x", i32) + val!(1);
        prop_assert_eq!(
            inp.then(Function::anonymous(arg_names, f)).to_string(),
            format!("let {:?} = {}; {}", arg_names, inp, f)
        );
    }
}
