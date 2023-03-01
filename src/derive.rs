pub(crate) use paste::paste;

macro_rules! derive_try_from_bounded_float {
    ( $from:tt, $into:tt ) => {
        crate::derive::paste! {
            #[doc = "Error returned when '" $into "' is given an invalid value."]
            #[derive(Clone, Copy, Debug, derive_more::Display, PartialEq, Eq)]
            pub enum [<$into TryFromError>] {
                /// Value is NaN.
                IsNan,
                /// Value is below the lower bound.
                TooLow,
                /// Value is above the upper bound.
                TooHigh,
            }

            impl core::convert::TryFrom<$from> for $into {
                type Error = [<$into TryFromError>];

                fn try_from(value: $from) -> Result<Self, Self::Error> {
                    if value.is_nan() {
                        Err(Self::Error::IsNan)
                    } else if value < <Self as LowerBounded>::min_value().into() {
                        Err(Self::Error::TooLow)
                    } else if value > <Self as UpperBounded>::max_value().into() {
                        Err(Self::Error::TooHigh)
                    } else {
                        Ok(Self(value))
                    }
                }
            }
        }
    };
}

macro_rules! derive_try_from_lower_bounded {
    ( $from:tt, $into:tt ) => {
        crate::derive::paste! {
            #[doc = "Error returned when '" $into "' is given a value below the lower bound."]
            #[derive(Clone, Copy, Debug, derive_more::Display)]
            pub struct [<$into TryFromError>];

            impl core::convert::TryFrom<$from> for $into {
                type Error = [<$into TryFromError>];

                fn try_from(value: $from) -> Result<Self, Self::Error> {
                    if value < <Self as LowerBounded>::min_value().into() {
                        Err([<$into TryFromError>])
                    } else {
                        Ok(Self(value))
                    }
                }
            }
        }
    };
}

macro_rules! derive_from_str_from_try_into {
    ( $from:tt, $into:tt ) => {
        crate::derive::paste! {
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

pub(crate) use derive_from_str_from_try_into;
pub(crate) use derive_try_from_bounded_float;
pub(crate) use derive_try_from_lower_bounded;
