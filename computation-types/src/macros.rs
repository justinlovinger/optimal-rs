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
