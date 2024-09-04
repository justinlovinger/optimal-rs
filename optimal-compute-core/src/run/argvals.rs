use core::fmt;
use std::{
    any::type_name,
    collections::BTreeMap,
    ops::{Deref, DerefMut},
};

use downcast_rs::{impl_downcast, Downcast};
use paste::paste;

use crate::{Args, Name};

pub trait ArgVal: Downcast + fmt::Debug {
    fn boxed_clone(&self) -> Box<dyn ArgVal>;
}
impl_downcast!(ArgVal);
impl<T> ArgVal for T
where
    T: 'static + Clone + fmt::Debug,
{
    fn boxed_clone(&self) -> Box<dyn ArgVal> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn ArgVal> {
    fn clone(&self) -> Self {
        // Calling `boxed_clone` without `as_ref`
        // will result in a stack overflow.
        self.as_ref().boxed_clone()
    }
}

#[derive(Clone, Debug)]
pub struct ArgVals(BTreeMap<Name, Box<dyn ArgVal>>);

impl Default for ArgVals {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! argvals {
( ) => {
    $crate::run::ArgVals::new()
};
( ($name:expr, $val:expr) ) => {
    $crate::run::ArgVals::singleton($name, $val)
};
( ($name:expr, $val:expr), $( $rest:tt ),* ) => {
    $crate::run::ArgVals::singleton($name, $val).union(argvals![$( $rest ),*])
};
}

impl ArgVals {
    pub fn new() -> Self {
        ArgVals(BTreeMap::new())
    }

    pub fn singleton<T>(name: Name, value: T) -> Self
    where
        T: ArgVal,
    {
        let value = Box::new(value) as Box<dyn ArgVal>;
        ArgVals(std::iter::once((name, value)).collect())
    }

    pub fn union(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    /// Return a (map with `fst_args`, map with `snd_args`) tuple if all arguments are present.
    pub fn partition(self, fst_args: &Args, snd_args: &Args) -> Result<(Self, Self), PartitionErr> {
        // We want to avoid making a new map
        // if we can simply return this map
        // as the partition with all arguments.
        if snd_args.is_empty() && fst_args.len() == self.0.len() {
            if fst_args.iter().all(|arg| self.0.contains_key(arg)) {
                Ok((self, Self::new()))
            } else {
                Err(PartitionErr::Missing(
                    fst_args
                        .iter()
                        .find(|arg| !self.0.contains_key(arg))
                        .unwrap(),
                ))
            }
        } else if fst_args.is_empty() && snd_args.len() == self.0.len() {
            if snd_args.iter().all(|arg| self.0.contains_key(arg)) {
                Ok((Self::new(), self))
            } else {
                Err(PartitionErr::Missing(
                    snd_args
                        .iter()
                        .find(|arg| !self.0.contains_key(arg))
                        .unwrap(),
                ))
            }
        } else {
            let mut fst_map = Self::new();
            let mut snd_map = Self::new();

            for (arg, val) in self.0.into_iter() {
                match (fst_args.contains(arg), snd_args.contains(arg)) {
                    (true, true) => {
                        fst_map.0.insert(arg, val.clone());
                        snd_map.0.insert(arg, val);
                    }
                    (true, false) => {
                        fst_map.0.insert(arg, val);
                    }
                    (false, true) => {
                        snd_map.0.insert(arg, val);
                    }
                    (false, false) => {}
                }
            }

            if fst_args.iter().all(|arg| fst_map.0.contains_key(arg))
                && snd_args.iter().all(|arg| snd_map.0.contains_key(arg))
            {
                Ok((fst_map, snd_map))
            } else {
                Err(PartitionErr::Missing(
                    fst_args
                        .iter()
                        .find(|arg| !fst_map.0.contains_key(arg))
                        .unwrap_or_else(|| {
                            snd_args
                                .iter()
                                .find(|arg| !snd_map.0.contains_key(arg))
                                .unwrap()
                        }),
                ))
            }
        }
    }

    /// Return the given argument
    /// if it is present
    /// and has the right type.
    ///
    /// Note,
    /// if argument is present
    /// but has the wrong type,
    /// it will still be removed.
    pub fn pop<T>(&mut self, name: Name) -> Result<T, PopErr>
    where
        T: 'static + ArgVal,
    {
        self.0
            .remove(name)
            .map_or_else(
                || Err(PopErr::Missing(name)),
                |x| {
                    x.downcast().map_err(|x| PopErr::WrongType {
                        name,
                        value: x,
                        ty: type_name::<T>(),
                    })
                },
            )
            .map(|x: Box<T>| *x)
    }

    #[cfg(test)]
    fn into_map<T>(self) -> BTreeMap<Name, T>
    where
        T: 'static + ArgVal,
    {
        self.0
            .into_iter()
            .map(|(k, v)| (k, *v.downcast().unwrap_or_else(|_| panic!())))
            .collect()
    }
}

macro_rules! impl_partition_n {
    ( $n:expr, $( $i:expr ),* ) => {
        paste! {
            impl ArgVals {
                /// Return a tuple with each requested set of arguments
                /// if all arguments are present.
                #[allow(clippy::too_many_arguments)]
                pub fn [<partition $n>](mut self, $( [<args_ $i>]: &Args ),* ) -> Result<( $( impl_partition_n!(@as_self $i) ),* ), PartitionErr> {
                    let all_args = [ $( [<args_ $i>] ),* ];

                    let mut partitions = Vec::new();
                    for i in 0..($n - 1) {
                        let (next, rest) = self.partition(all_args[i], &Args::from_args(all_args.into_iter().skip(i + 1)))?;
                        partitions.push(next);
                        self = rest;
                    }
                    partitions.push(self);

                    // My hope is
                    // the Rust compiler will understand
                    // it does not actually need to clone.
                    Ok(( $( partitions[$i].clone() ),* ))
                }
            }
        }
    };
    ( @as_self $i:expr ) => {
        Self
    };
}

impl_partition_n!(3, 0, 1, 2);
impl_partition_n!(4, 0, 1, 2, 3);
impl_partition_n!(5, 0, 1, 2, 3, 4);
impl_partition_n!(6, 0, 1, 2, 3, 4, 5);
impl_partition_n!(7, 0, 1, 2, 3, 4, 5, 6);
impl_partition_n!(8, 0, 1, 2, 3, 4, 5, 6, 7);
impl_partition_n!(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_partition_n!(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_partition_n!(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_partition_n!(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_partition_n!(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_partition_n!(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_partition_n!(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_partition_n!(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Value<T>(pub T);

impl<T> PartialEq<T> for Value<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &T) -> bool {
        self.0.eq(other)
    }
}

impl<T> PartialOrd<T> for Value<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl<T> Deref for Value<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Value<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait FromArgsVals<Args, Vals> {
    fn from_args_vals(args: Args, vals: Vals) -> Self;
}

impl<Names0, T0> FromArgsVals<(Names0,), (T0,)> for ArgVals
where
    ArgVals: FromArgsVals<Names0, T0>,
{
    fn from_args_vals(args: (Names0,), vals: (T0,)) -> Self {
        Self::from_args_vals(args.0, vals.0)
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum PartitionErr {
    #[error("`ArgVals` is missing `{0}`.")]
    Missing(&'static str),
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum PopErr {
    #[error("`ArgVals` is missing `{0}`.")]
    Missing(&'static str),
    #[error("Expected type `{ty}` for `{name}`, found value `{value:?}`.")]
    WrongType {
        name: &'static str,
        value: Box<dyn ArgVal>,
        ty: &'static str,
    },
}

macro_rules! impl_from_args_vals {
    ( $( $i:expr ),* ) => {
        paste! {
            impl< $( [<Names $i>] ),* , $( [<T $i>] ),* > FromArgsVals<( $( [<Names $i>] ),* ), ( $( [<T $i>] ),* )> for ArgVals
            where
                $( ArgVals: FromArgsVals<[<Names $i>], [<T $i>]> ),*
            {
                fn from_args_vals(args: ( $( [<Names $i>] ),* ), vals: ( $( [<T $i>] ),* )) -> Self {
                    let mut out = ArgVals::new();
                    $(
                        out = out.union(Self::from_args_vals(args.$i, vals.$i));
                    )*
                    out
                }
            }
        }
    };
    ( @as_name $i:expr ) => {
        Name
    };
}

impl_from_args_vals!(0, 1);
impl_from_args_vals!(0, 1, 2);
impl_from_args_vals!(0, 1, 2, 3);
impl_from_args_vals!(0, 1, 2, 3, 4);
impl_from_args_vals!(0, 1, 2, 3, 4, 5);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_from_args_vals!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

impl<T> FromArgsVals<Name, Value<T>> for ArgVals
where
    T: ArgVal,
{
    fn from_args_vals(args: Name, vals: Value<T>) -> Self {
        Self::singleton(args, vals.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::args;

    #[test]
    fn partition_should_return_full_map_for_fst_when_all_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert_eq!(
            arg_vals
                .clone()
                .partition(&args!["foo", "bar"], &args![])
                .unwrap()
                .0
                .into_map::<i32>(),
            arg_vals.into_map()
        );
    }

    #[test]
    fn partition_should_return_full_map_for_snd_when_all_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert_eq!(
            arg_vals
                .clone()
                .partition(&args![], &args!["foo", "bar"])
                .unwrap()
                .1
                .into_map::<i32>(),
            arg_vals.into_map()
        );
    }

    #[test]
    fn partition_should_return_maps_with_given_args_when_some_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2), ("baz", 3)];
        let (left, right) = arg_vals.partition(&args!["baz"], &args!["foo"]).unwrap();
        assert_eq!(left.into_map::<i32>(), argvals![("baz", 3)].into_map(),);
        assert_eq!(right.into_map::<i32>(), argvals![("foo", 1)].into_map(),);
    }

    #[test]
    fn partition_should_duplicate_args_required_by_both() {
        let arg_vals = argvals![("foo", 1), ("bar", 2), ("baz", 3), ("biz", 4)];
        let (left, right) = arg_vals
            .partition(&args!["foo", "bar", "baz"], &args!["foo", "bar", "biz"])
            .unwrap();
        assert_eq!(
            left.into_map::<i32>(),
            argvals![("foo", 1), ("bar", 2), ("baz", 3)].into_map(),
        );
        assert_eq!(
            right.into_map::<i32>(),
            argvals![("foo", 1), ("bar", 2), ("biz", 4)].into_map(),
        );
    }

    #[test]
    fn partition_should_return_none_if_arg_missing() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert!(arg_vals
            .clone()
            .partition(&args![], &args!["foo", "baz"])
            .is_err());
        assert!(arg_vals.partition(&args!["foo", "baz"], &args![]).is_err());
    }

    #[test]
    fn partition3_should_return_full_map_for_fst_when_all_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert_eq!(
            arg_vals
                .clone()
                .partition3(&args!["foo", "bar"], &args![], &args![])
                .unwrap()
                .0
                .into_map::<i32>(),
            arg_vals.into_map()
        );
    }

    #[test]
    fn partition3_should_return_full_map_for_snd_when_all_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert_eq!(
            arg_vals
                .clone()
                .partition3(&args![], &args!["foo", "bar"], &args![])
                .unwrap()
                .1
                .into_map::<i32>(),
            arg_vals.into_map()
        );
    }

    #[test]
    fn partition3_should_return_full_map_for_third_when_all_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert_eq!(
            arg_vals
                .clone()
                .partition3(&args![], &args![], &args!["foo", "bar"])
                .unwrap()
                .2
                .into_map::<i32>(),
            arg_vals.into_map()
        );
    }

    #[test]
    fn partition3_should_return_maps_with_given_args_when_some_args() {
        let arg_vals = argvals![("foo", 1), ("bar", 2), ("baz", 3), ("bin", 4)];
        let (fst, snd, third) = arg_vals
            .partition3(&args!["baz"], &args!["foo"], &args!["bar"])
            .unwrap();
        assert_eq!(fst.into_map::<i32>(), argvals![("baz", 3)].into_map(),);
        assert_eq!(snd.into_map::<i32>(), argvals![("foo", 1)].into_map(),);
        assert_eq!(third.into_map::<i32>(), argvals![("bar", 2)].into_map(),);
    }

    #[test]
    fn partition3_should_duplicate_args_required_by_more_than_one() {
        let arg_vals = argvals![("foo", 1), ("bar", 2), ("baz", 3), ("biz", 4), ("bop", 5)];
        let (fst, snd, third) = arg_vals
            .partition3(
                &args!["foo", "bar", "baz"],
                &args!["foo", "bar", "biz"],
                &args!["bar", "biz", "bop"],
            )
            .unwrap();
        assert_eq!(
            fst.into_map::<i32>(),
            argvals![("foo", 1), ("bar", 2), ("baz", 3)].into_map(),
        );
        assert_eq!(
            snd.into_map::<i32>(),
            argvals![("foo", 1), ("bar", 2), ("biz", 4)].into_map(),
        );
        assert_eq!(
            third.into_map::<i32>(),
            argvals![("bar", 2), ("biz", 4), ("bop", 5)].into_map(),
        );
    }

    #[test]
    fn partition3_should_return_none_if_arg_missing() {
        let arg_vals = argvals![("foo", 1), ("bar", 2)];
        assert!(arg_vals
            .clone()
            .partition3(&args!["foo", "baz"], &args![], &args![])
            .is_err());
        assert!(arg_vals
            .clone()
            .partition3(&args![], &args!["foo", "baz"], &args![])
            .is_err());
        assert!(arg_vals
            .partition3(&args![], &args![], &args!["foo", "baz"])
            .is_err());
    }

    #[test]
    fn pop_should_return_value_if_available_and_correct_type() {
        let mut arg_vals = argvals![("foo", 1), ("bar", 2.0)];
        assert_eq!(arg_vals.pop::<i32>("foo").ok(), Some(1));
        assert_eq!(arg_vals.pop::<f64>("bar").ok(), Some(2.0));
    }

    #[test]
    fn pop_should_return_none_if_not_available() {
        let mut arg_vals = argvals![("foo", 1), ("bar", 2.0)];
        assert_eq!(arg_vals.pop::<i32>("baz").ok(), None);
    }

    #[test]
    fn pop_should_return_none_if_wrong_type() {
        let mut arg_vals = argvals![("foo", 1), ("bar", 2.0)];
        assert_eq!(arg_vals.pop::<f64>("foo").ok(), None);
    }
}
