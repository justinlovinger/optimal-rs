use derive_more::Display;

macro_rules! derive_try_from_bounded_float {
    ( $from:tt, $into:tt ) => {
        impl core::convert::TryFrom<$from> for $into {
            type Error = crate::BoundedFloatError;

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
    };
}

macro_rules! derive_try_from_lower_bounded {
    ( $from:tt, $into:tt ) => {
        impl core::convert::TryFrom<$from> for $into {
            type Error = crate::LowerBoundedError;

            fn try_from(value: $from) -> Result<Self, Self::Error> {
                if value < <Self as LowerBounded>::min_value().into() {
                    Err(crate::LowerBoundedError)
                } else {
                    Ok(Self(value))
                }
            }
        }
    };
}

macro_rules! derive_from_str_from_try_into {
    ( $from:tt, $into:tt ) => {
        impl std::str::FromStr for $into {
            type Err = crate::FromStrFromTryIntoError<
                <$from as std::str::FromStr>::Err,
                <Self as TryFrom<$from>>::Error,
            >;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                s.parse::<$from>()
                    .map_err(|e| Self::Err::FromStr(e))
                    .and_then(|x| x.try_into().map_err(Self::Err::TryInto))
            }
        }
    };
}

pub(crate) use derive_from_str_from_try_into;
pub(crate) use derive_try_from_bounded_float;
pub(crate) use derive_try_from_lower_bounded;

/// Error returned when a bounded float is given an invalid value.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum BoundedFloatError {
    /// Value is NaN.
    IsNan,
    /// Value is below the lower bound.
    TooLow,
    /// Value is above the upper bound.
    TooHigh,
}

/// Error returned when a bounded number is given an invalid value.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum BoundedError {
    /// Value is below the lower bound.
    TooLow,
    /// Value is above the upper bound.
    TooHigh,
}

/// Error returned when a lower bounded number is given a value below the lower bound.
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub struct LowerBoundedError;

/// Error returned when failing to convert from a string or into the resulting type.
#[derive(Clone, Copy, Debug)]
pub enum FromStrFromTryIntoError<A, B> {
    /// Error convering to the intermediate type.
    FromStr(A),
    /// Error convering to the resulting type.
    TryInto(B),
}
