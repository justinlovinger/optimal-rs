#![allow(unused_macros)]
#![allow(unused_imports)]

//! Macros for implementing bounded number types.

pub use paste;
pub use thiserror;

#[macro_export]
macro_rules! derive_new_from_bounded_partial_ord {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        $crate::_derive_new_from_bounded_partial_ord!(
            $type<$a: $bound>,
            $a,
            IsIncomparable,
            "incomparable"
        );
    };
    ( $type:ident {( $inner:ty )} ) => {
        $crate::_derive_new_from_bounded_partial_ord!(
            $type,
            $inner,
            IsIncomparable,
            "incomparable"
        );
    };
}

#[macro_export]
macro_rules! derive_new_from_bounded_float {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        $crate::_derive_new_from_bounded_partial_ord!($type<$a: $bound>, $a, IsNan, "NaN");
    };
    ( $type:ident ( $inner:ty ) ) => {
        $crate::_derive_new_from_bounded_partial_ord!($type, $inner, IsNan, "NaN");
    };
}

#[macro_export]
macro_rules! _derive_new_from_bounded_partial_ord {
    ( $type:ident $( < $a:ty : $bound:ident > )?, $inner:ty, $incomparable_name:ident, $incomparable_str:literal ) => {
        $crate::paste::paste! {
            #[doc = "Error returned when '" $type "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, $crate::thiserror::Error, PartialEq)]
            pub enum [<Invalid $type Error>] $(< $a : $bound >)? {
                #[doc = "Value is " $incomparable_str "."]
                #[error("{0} is {}", $incomparable_str)]
                $incomparable_name($inner),
                /// Value is below lower bound.
                #[error("{0} is below lower bound ({})", < $type $(< $a >)? > ::min_value())]
                TooLow($inner),
                /// Value is above upper bound.
                #[error("{0} is above upper bound ({})", < $type $(< $a >)? > ::max_value())]
                TooHigh($inner),
            }

            impl $(< $a : $bound >)? $type $(< $a >)? {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>]  $(< $a >)? > {
                    match (
                        Self(value).partial_cmp(&Self::min_value()),
                        Self(value).partial_cmp(&Self::max_value()),
                    ) {
                        (None, _) | (_, None) => Err([<Invalid $type Error>]::$incomparable_name(value)),
                        (Some(std::cmp::Ordering::Less), _) => Err([<Invalid $type Error>]::TooLow(value)),
                        (_, Some(std::cmp::Ordering::Greater)) => Err([<Invalid $type Error>]::TooHigh(value)),
                        _ => Ok(Self(value)),
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! derive_new_from_lower_bounded_partial_ord {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        $crate::_derive_new_from_lower_bounded_partial_ord!(
            $type<$a: $bound>,
            $a,
            IsIncomparable,
            "incomparable"
        );
    };
    ( $type:ident {( $inner:ty )} ) => {
        $crate::_derive_new_from_lower_bounded_partial_ord!(
            $type,
            $inner,
            IsIncomparable,
            "incomparable"
        );
    };
}

#[macro_export]
macro_rules! derive_new_from_lower_bounded_float {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        $crate::_derive_new_from_lower_bounded_partial_ord!($type<$a: $bound>, $a, IsNan, "NaN");
    };
    ( $type:ident ( $inner:ty ) ) => {
        $crate::_derive_new_from_lower_bounded_partial_ord!($type, $inner, IsNan, "NaN");
    };
}

#[macro_export]
macro_rules! _derive_new_from_lower_bounded_partial_ord {
    ( $type:ident $( < $a:ty : $bound:ident > )?, $inner:ty, $incomparable_name:ident, $incomparable_str:literal ) => {
        $crate::paste::paste! {
            #[doc = "Error returned when '" $type "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, $crate::thiserror::Error, PartialEq, Eq)]
            pub enum [<Invalid $type Error>] $(< $a : $bound >)? {
                #[doc = "Value is " $incomparable_str "."]
                #[error("{0} is {}", $incomparable_str)]
                $incomparable_name($inner),
                /// Value is below lower bound.
                #[error("{0} is below lower bound ({})", < $type $(< $a >)? > ::min_value())]
                TooLow($inner),
            }

            impl $(< $a : $bound >)? $type $(< $a >)? {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>] $(< $a >)? > {
                    match Self(value).partial_cmp(&Self::min_value()) {
                        None => Err([<Invalid $type Error>]::$incomparable_name(value)),
                        Some(std::cmp::Ordering::Less) => Err([<Invalid $type Error>]::TooLow(value)),
                        _ => Ok(Self(value)),
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! derive_new_from_lower_bounded {
    ( $type:ident ( $inner: ty ) ) => {
        $crate::paste::paste! {
            #[doc = "Error returned when '" $type "' is given a value below lower bound."]
            #[derive(Clone, Copy, Debug, $crate::thiserror::Error)]
            #[error("{0} is below lower bound ({})", $type::min_value())]
            pub struct [<Invalid $type Error>]($inner);

            impl $type {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>]> {
                    if Self(value) < Self::min_value() {
                        Err([<Invalid $type Error>](value))
                    } else {
                        Ok(Self(value))
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! derive_try_from_from_new {
    ( $type:ident ( $inner:ty ) ) => {
        $crate::paste::paste! {
            impl core::convert::TryFrom<$inner> for $type {
                type Error = [<Invalid $type Error>];
                fn try_from(value: $inner) -> Result<Self, Self::Error> {
                    $type::new(value)
                }
            }
        }
    };
}

#[macro_export]
macro_rules! derive_from_str_from_try_into {
    ( $type:ident ( $inner:ty ) ) => {
        $crate::paste::paste! {
            #[doc = "Error returned when failing to convert from a string or into '" $type "'."]
            #[derive(Clone, Debug, $crate::thiserror::Error)]
            #[error(transparent)]
            pub struct [<$type FromStrError>](#[from] [<_ $type FromStrError>]);

            #[derive(Clone, Debug, $crate::thiserror::Error)]
            enum [<_ $type FromStrError>] {
                #[doc = "Error convering from 'str' to '" $inner "'."]
                #[error("Failed to convert from 'str': {0}")]
                FromStr(<$inner as std::str::FromStr>::Err),
                #[doc = "Error convering from '" $inner "' to '" $type "'."]
                #[error("Failed to convert into type: {0}")]
                TryInto(<$type as TryFrom<$inner>>::Error),
            }

            impl std::str::FromStr for $type {
                type Err = [<$type FromStrError>];

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    s.parse::<$inner>()
                        .map_err([<_ $type FromStrError>]::FromStr)
                        .and_then(|x| x.try_into().map_err([<_ $type FromStrError>]::TryInto))
                        .map_err(|x| x.into())
                }
            }
        }
    };
}

#[macro_export]
macro_rules! derive_into_inner {
    ( $type:ident ( $inner:ty ) ) => {
        $crate::paste::paste! {
            impl $type {
                #[doc = "Unwrap '" $type "' into inner value."]
                pub fn into_inner(self) -> $inner {
                    self.0
                }
            }
        }
    };
    ( $type:ident < $a:ty > ) => {
        $crate::paste::paste! {
            impl < $a > $type < $a > {
                #[doc = "Unwrap '" $type "' into inner value."]
                pub fn into_inner(self) -> $a {
                    self.0
                }
            }
        }
    };
}
