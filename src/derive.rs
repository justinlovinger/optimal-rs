macro_rules! derive_new_from_bounded_float {
    ( $from:tt, $into:tt ) => {
        paste::paste! {
            #[doc = "Error returned when '" $into "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, derive_more::Display, PartialEq, Eq)]
            pub enum [<Invalid $into Error>] {
                /// Value is NaN.
                IsNan,
                /// Value is below the lower bound.
                TooLow,
                /// Value is above the upper bound.
                TooHigh,
            }

            impl $into {
                #[doc = "Return a new '" $into "' if given a valid value."]
                pub fn new(value: $from) -> Result<Self, [<Invalid $into Error>]> {
                    match (
                        Self(value).partial_cmp(&Self::min_value()),
                        Self(value).partial_cmp(&Self::max_value()),
                    ) {
                        (None, _) | (_, None) => Err([<Invalid $into Error>]::IsNan),
                        (Some(std::cmp::Ordering::Less), _) => Err([<Invalid $into Error>]::TooLow),
                        (_, Some(std::cmp::Ordering::Greater)) => Err([<Invalid $into Error>]::TooHigh),
                        _ => Ok(Self(value)),
                    }
                }
            }
        }
    };
}

macro_rules! derive_new_from_lower_bounded {
    ( $from:tt, $into:tt ) => {
        paste::paste! {
            #[doc = "Error returned when '" $into "' is given a value below the lower bound."]
            #[derive(Clone, Copy, Debug, derive_more::Display)]
            pub struct [<Invalid $into Error>];

            impl $into {
                #[doc = "Return a new '" $into "' if given a valid value."]
                pub fn new(value: $from) -> Result<Self, [<Invalid $into Error>]> {
                    if Self(value) < Self::min_value() {
                        Err([<Invalid $into Error>])
                    } else {
                        Ok(Self(value))
                    }
                }
            }
        }
    };
}

macro_rules! derive_try_from_from_new {
    ( $from:tt, $into:tt ) => {
        paste::paste! {
            impl core::convert::TryFrom<$from> for $into {
                type Error = [<Invalid $into Error>];
                fn try_from(value: $from) -> Result<Self, Self::Error> {
                    $into::new(value)
                }
            }
        }
    };
}

macro_rules! derive_from_str_from_try_into {
    ( $from:tt, $into:tt ) => {
        paste::paste! {
            #[doc = "Error returned when failing to convert from a string or into '" $into "'."]
            #[derive(Clone, Copy, Debug)]
            pub enum [<$into FromStrError>]<A, B> {
                #[doc = "Error convering to '" $from "'."]
                FromStr(A),
                #[doc = "Error convering to '" $into "'."]
                TryInto(B),
            }

            impl std::str::FromStr for $into {
                type Err = [<$into FromStrError>]<
                    <$from as std::str::FromStr>::Err,
                    <Self as TryFrom<$from>>::Error,
                >;

                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    s.parse::<$from>()
                        .map_err(|e| Self::Err::FromStr(e))
                        .and_then(|x| x.try_into().map_err(Self::Err::TryInto))
                }
            }
        }
    };
}

macro_rules! derive_into_inner {
    ( $type:tt, $inner:tt ) => {
        paste::paste! {
            impl $type {
                #[doc = "Unwrap '" $type "' into inner value."]
                pub fn into_inner(self) -> $inner {
                    self.0
                }
            }
        }
    };
}

pub(crate) use derive_from_str_from_try_into;
pub(crate) use derive_into_inner;
pub(crate) use derive_new_from_bounded_float;
pub(crate) use derive_new_from_lower_bounded;
pub(crate) use derive_try_from_from_new;
