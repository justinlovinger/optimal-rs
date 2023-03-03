#![allow(unused_macros)]
#![allow(unused_imports)]

macro_rules! derive_new_from_bounded_partial_ord {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        crate::derive::_derive_new_from_bounded_partial_ord!(
            $type<$a: $bound>,
            $a,
            IsIncomparable,
            "incomparable"
        );
    };
    ( $type:ident {( $inner:ty )} ) => {
        crate::derive::_derive_new_from_bounded_partial_ord!(
            $type,
            $inner,
            IsIncomparable,
            "incomparable"
        );
    };
}

macro_rules! derive_new_from_bounded_float {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        crate::derive::_derive_new_from_bounded_partial_ord!($type<$a: $bound>, $a, IsNan, "NaN");
    };
    ( $type:ident ( $inner:ty ) ) => {
        crate::derive::_derive_new_from_bounded_partial_ord!($type, $inner, IsNan, "NaN");
    };
}

macro_rules! _derive_new_from_bounded_partial_ord {
    ( $type:ident $( < $a:ty : $bound:ident > )?, $inner:ty, $incomparable_name:ident, $incomparable_str:literal ) => {
        paste::paste! {
            #[doc = "Error returned when '" $type "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, derive_more::Display, PartialEq, Eq)]
            pub enum [<Invalid $type Error>] {
                #[doc = "Value is " $incomparable_str "."]
                $incomparable_name,
                /// Value is below the lower bound.
                TooLow,
                /// Value is above the upper bound.
                TooHigh,
            }

            impl $(< $a : $bound >)? $type $(< $a >)? {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>]> {
                    match (
                        Self(value).partial_cmp(&Self::min_value()),
                        Self(value).partial_cmp(&Self::max_value()),
                    ) {
                        (None, _) | (_, None) => Err([<Invalid $type Error>]::$incomparable_name),
                        (Some(std::cmp::Ordering::Less), _) => Err([<Invalid $type Error>]::TooLow),
                        (_, Some(std::cmp::Ordering::Greater)) => Err([<Invalid $type Error>]::TooHigh),
                        _ => Ok(Self(value)),
                    }
                }
            }
        }
    };
}

macro_rules! derive_new_from_lower_bounded_partial_ord {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        crate::derive::_derive_new_from_lower_bounded_partial_ord!(
            $type<$a: $bound>,
            $a,
            IsIncomparable,
            "incomparable"
        );
    };
    ( $type:ident {( $inner:ty )} ) => {
        crate::derive::_derive_new_from_lower_bounded_partial_ord!(
            $type,
            $inner,
            IsIncomparable,
            "incomparable"
        );
    };
}

macro_rules! derive_new_from_lower_bounded_float {
    ( $type:ident < $a:ty : $bound:ident > ) => {
        crate::derive::_derive_new_from_lower_bounded_partial_ord!(
            $type<$a: $bound>,
            $a,
            IsNan,
            "NaN"
        );
    };
    ( $type:ident ( $inner:ty ) ) => {
        crate::derive::_derive_new_from_lower_bounded_partial_ord!($type, $inner, IsNan, "NaN");
    };
}

macro_rules! _derive_new_from_lower_bounded_partial_ord {
    ( $type:ident $( < $a:ty : $bound:ident > )?, $inner:ty, $incomparable_name:ident, $incomparable_str:literal ) => {
        paste::paste! {
            #[doc = "Error returned when '" $type "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, derive_more::Display, PartialEq, Eq)]
            pub enum [<Invalid $type Error>] {
                #[doc = "Value is " $incomparable_str "."]
                $incomparable_name,
                /// Value is below the lower bound.
                TooLow,
            }

            impl $(< $a : $bound >)? $type $(< $a >)? {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>]> {
                    match Self(value).partial_cmp(&Self::min_value()) {
                        None => Err([<Invalid $type Error>]::$incomparable_name),
                        Some(std::cmp::Ordering::Less) => Err([<Invalid $type Error>]::TooLow),
                        _ => Ok(Self(value)),
                    }
                }
            }
        }
    };
}

macro_rules! derive_new_from_lower_bounded {
    ( $type:ident ( $inner: ty ) ) => {
        paste::paste! {
            #[doc = "Error returned when '" $type "' is given a value below the lower bound."]
            #[derive(Clone, Copy, Debug, derive_more::Display)]
            pub struct [<Invalid $type Error>];

            impl $type {
                #[doc = "Return a new '" $type "' if given a valid value."]
                pub fn new(value: $inner) -> Result<Self, [<Invalid $type Error>]> {
                    if Self(value) < Self::min_value() {
                        Err([<Invalid $type Error>])
                    } else {
                        Ok(Self(value))
                    }
                }
            }
        }
    };
}

macro_rules! derive_try_from_from_new {
    ( $type:ident ( $inner:ty ) ) => {
        paste::paste! {
            impl core::convert::TryFrom<$inner> for $type {
                type Error = [<Invalid $type Error>];
                fn try_from(value: $inner) -> Result<Self, Self::Error> {
                    $type::new(value)
                }
            }
        }
    };
}

macro_rules! derive_from_str_from_try_into {
    ( $type:ident ( $inner:ty ) ) => {
        paste::paste! {
            #[doc = "Error returned when failing to convert from a string or into '" $type "'."]
            #[derive(Clone, Copy, Debug)]
            pub enum [<$type FromStrError>]<A, B> {
                #[doc = "Error convering to '" $inner "'."]
                FromStr(A),
                #[doc = "Error convering to '" $type "'."]
                TryInto(B),
            }

            impl std::str::FromStr for $type {
                type Err = [<$type FromStrError>]<
                    <$inner as std::str::FromStr>::Err,
                    <Self as TryFrom<$inner>>::Error,
                >;

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    s.parse::<$inner>()
                        .map_err(|e| Self::Err::FromStr(e))
                        .and_then(|x| x.try_into().map_err(Self::Err::TryInto))
                }
            }
        }
    };
}

macro_rules! derive_into_inner {
    ( $type:ident ( $inner:ty ) ) => {
        paste::paste! {
            impl $type {
                #[doc = "Unwrap '" $type "' into inner value."]
                pub fn into_inner(self) -> $inner {
                    self.0
                }
            }
        }
    };
    ( $type:ident < $a:ty > ) => {
        paste::paste! {
            impl < $a > $type < $a > {
                #[doc = "Unwrap '" $type "' into inner value."]
                pub fn into_inner(self) -> $a {
                    self.0
                }
            }
        }
    };
}

pub(crate) use _derive_new_from_bounded_partial_ord;
pub(crate) use _derive_new_from_lower_bounded_partial_ord;
pub(crate) use derive_from_str_from_try_into;
pub(crate) use derive_into_inner;
pub(crate) use derive_new_from_bounded_float;
pub(crate) use derive_new_from_bounded_partial_ord;
pub(crate) use derive_new_from_lower_bounded;
pub(crate) use derive_new_from_lower_bounded_float;
pub(crate) use derive_new_from_lower_bounded_partial_ord;
pub(crate) use derive_try_from_from_new;
