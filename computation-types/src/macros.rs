pub use paste;

#[macro_export]
macro_rules! impl_core_ops {
    ( $ty:ident $( < $( $gen:ident ),* > )? ) => {
        impl_core_ops!(@impl Add for $ty $( < $( $gen ),* > )?);
        impl_core_ops!(@impl Sub for $ty $( < $( $gen ),* > )?);
        impl_core_ops!(@impl Mul for $ty $( < $( $gen ),* > )?);
        impl_core_ops!(@impl Div for $ty $( < $( $gen ),* > )?);

        impl $( < $( $gen ),* > )? core::ops::Neg for $ty $( < $( $gen ),* > )?
        where
            Self: Computation,
            $crate::math::Neg<Self>: $crate::Computation,
        {
            type Output = $crate::math::Neg<Self>;

            fn neg(self) -> Self::Output {
                <Self as $crate::Computation>::neg(self)
            }
        }
    };
    ( @impl $op:ident for $ty:ident $( < $( $gen:ident ),* > )? ) => {
        $crate::macros::paste::paste! {
            impl<Rhs, $( $( $gen ),* )?> core::ops::$op<Rhs> for $ty $( < $( $gen ),* > )?
            where
                Self: Computation,
                $crate::math::$op<Self, Rhs>: $crate::Computation,
            {
                type Output = $crate::math::$op<Self, Rhs>;

                fn [<$op:lower>](self, rhs: Rhs) -> Self::Output {
                    <Self as $crate::Computation>::[<$op:lower>](self, rhs)
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_display_for_inline_binary {
    ( $op:ident, $inline:literal ) => {
        impl<A, B> core::fmt::Display for $op<A, B>
        where
            Self: $crate::Computation,
            A: core::fmt::Display,
            B: core::fmt::Display,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "({} {} {})", self.0, $inline, self.1)
            }
        }
    };
}

#[macro_export]
macro_rules! impl_computation_fn_for_unary {
    ( $op:ident ) => {
        impl<A> ComputationFn for $op<A>
        where
            Self: Computation,
            A: ComputationFn,
            $op<A::Filled>: Computation,
        {
            type Filled = $op<A::Filled>;

            fn fill(self, named_args: NamedArgs) -> Self::Filled {
                $op(self.0.fill(named_args))
            }

            fn arg_names(&self) -> $crate::Names {
                self.0.arg_names()
            }
        }
    };
}

#[macro_export]
macro_rules! impl_computation_fn_for_binary {
    ( $op:ident ) => {
        impl<A, B> ComputationFn for $op<A, B>
        where
            Self: Computation,
            A: ComputationFn,
            B: ComputationFn,
            $op<A::Filled, B::Filled>: Computation,
        {
            type Filled = $op<A::Filled, B::Filled>;

            fn fill(self, named_args: NamedArgs) -> Self::Filled {
                let (args_0, args_1) = named_args
                    .partition(&self.0.arg_names(), &self.1.arg_names())
                    .unwrap_or_else(|e| panic!("{}", e));
                $op(self.0.fill(args_0), self.1.fill(args_1))
            }

            fn arg_names(&self) -> $crate::Names {
                self.0.arg_names().union(self.1.arg_names())
            }
        }
    };
}
