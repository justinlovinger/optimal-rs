use core::fmt;
use std::{any::type_name, collections::BTreeMap};

use downcast_rs::{impl_downcast, Downcast};
use paste::paste;

use crate::{Name, Names};

pub trait AnyArg: Downcast + fmt::Debug {
    fn boxed_clone(&self) -> Box<dyn AnyArg>;
}
impl_downcast!(AnyArg);
impl<T> AnyArg for T
where
    T: 'static + Clone + fmt::Debug,
{
    fn boxed_clone(&self) -> Box<dyn AnyArg> {
        Box::new(self.clone())
    }
}
impl Clone for Box<dyn AnyArg> {
    fn clone(&self) -> Self {
        // Calling `boxed_clone` without `as_ref`
        // will result in a stack overflow.
        self.as_ref().boxed_clone()
    }
}

#[derive(Clone, Debug)]
pub struct NamedArgs(BTreeMap<Name, Box<dyn AnyArg>>);

impl Default for NamedArgs {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! named_args {
( ) => {
    $crate::NamedArgs::new()
};
( ($name:expr, $arg:expr) ) => {
    $crate::NamedArgs::singleton($name, $arg)
};
( ($name:expr, $arg:expr), $( $rest:tt ),* ) => {
    $crate::NamedArgs::singleton($name, $arg).union(named_args![$( $rest ),*])
};
}

impl NamedArgs {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn singleton<T>(name: Name, arg: T) -> Self
    where
        T: AnyArg,
    {
        let arg = Box::new(arg) as Box<dyn AnyArg>;
        Self(std::iter::once((name, arg)).collect())
    }

    pub fn union(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    pub fn contains_args(&self, args: &Names) -> bool {
        args.iter().all(|arg| self.0.contains_key(arg))
    }

    /// Return a (map with `fst_names`, map with `snd_names`) tuple if all arguments are present.
    pub fn partition(
        self,
        fst_names: &Names,
        snd_names: &Names,
    ) -> Result<(Self, Self), PartitionErr> {
        // We want to avoid making a new map
        // if we can simply return this map
        // as the partition with all arguments.
        if snd_names.is_empty() && fst_names.len() == self.0.len() {
            if self.contains_args(fst_names) {
                Ok((self, Self::new()))
            } else {
                Err(PartitionErr::Missing(
                    fst_names
                        .iter()
                        .find(|arg| !self.0.contains_key(arg))
                        .unwrap(),
                ))
            }
        } else if fst_names.is_empty() && snd_names.len() == self.0.len() {
            if self.contains_args(snd_names) {
                Ok((Self::new(), self))
            } else {
                Err(PartitionErr::Missing(
                    snd_names
                        .iter()
                        .find(|arg| !self.0.contains_key(arg))
                        .unwrap(),
                ))
            }
        } else {
            let mut fst_map = Self::new();
            let mut snd_map = Self::new();

            for (name, arg) in self.0.into_iter() {
                match (fst_names.contains(name), snd_names.contains(name)) {
                    (true, true) => {
                        fst_map.0.insert(name, arg.clone());
                        snd_map.0.insert(name, arg);
                    }
                    (true, false) => {
                        fst_map.0.insert(name, arg);
                    }
                    (false, true) => {
                        snd_map.0.insert(name, arg);
                    }
                    (false, false) => {}
                }
            }

            if fst_map.contains_args(fst_names) && snd_map.contains_args(snd_names) {
                Ok((fst_map, snd_map))
            } else {
                Err(PartitionErr::Missing(
                    fst_names
                        .iter()
                        .find(|arg| !fst_map.0.contains_key(arg))
                        .unwrap_or_else(|| {
                            snd_names
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
        T: 'static + AnyArg,
    {
        self.0
            .remove(name)
            .map_or_else(
                || Err(PopErr::Missing(name)),
                |x| {
                    x.downcast().map_err(|x| PopErr::WrongType {
                        name,
                        arg: x,
                        ty: type_name::<T>(),
                    })
                },
            )
            .map(|x: Box<T>| *x)
    }

    /// This function is unstable and likely to change.
    ///
    /// Use at your own risk.
    pub fn insert_raw(&mut self, name: Name, arg: Box<dyn AnyArg>) {
        self.0.insert(name, arg);
    }

    #[cfg(test)]
    fn into_map<T>(self) -> BTreeMap<Name, T>
    where
        T: 'static + AnyArg,
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
            impl NamedArgs {
                /// Return a tuple with each requested set of arguments
                /// if all arguments are present.
                #[allow(clippy::too_many_arguments)]
                pub fn [<partition $n>](mut self, $( [<names_ $i>]: &Names ),* ) -> Result<( $( impl_partition_n!(@as_self $i) ),* ), PartitionErr> {
                    let all_names = [ $( [<names_ $i>] ),* ];

                    let mut partitions = Vec::new();
                    for i in 0..($n - 1) {
                        let (next, rest) = self.partition(all_names[i], &Names::union_many(all_names.into_iter().skip(i + 1)))?;
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

#[derive(Clone, Debug, thiserror::Error)]
pub enum PartitionErr {
    #[error("`NamedArgs` is missing `{0}`.")]
    Missing(&'static str),
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum PopErr {
    #[error("`NamedArgs` is missing `{0}`.")]
    Missing(&'static str),
    #[error("Expected type `{ty}` for `{name}`, found arg `{arg:?}`.")]
    WrongType {
        name: &'static str,
        arg: Box<dyn AnyArg>,
        ty: &'static str,
    },
}

pub trait FromNamesArgs<Names, Args> {
    fn from_names_args(names: Names, args: Args) -> Self;
}

impl<T> FromNamesArgs<Name, T> for NamedArgs
where
    T: AnyArg,
{
    default fn from_names_args(names: Name, args: T) -> Self {
        Self::singleton(names, args)
    }
}

impl<Names0, T0> FromNamesArgs<(Names0,), (T0,)> for NamedArgs
where
    NamedArgs: FromNamesArgs<Names0, T0>,
{
    fn from_names_args(names: (Names0,), args: (T0,)) -> Self {
        Self::from_names_args(names.0, args.0)
    }
}

macro_rules! impl_from_names_args {
    ( $( $i:expr ),* ) => {
        paste! {
            impl< $( [<Names $i>] ),* , $( [<T $i>] ),* > FromNamesArgs<( $( [<Names $i>] ),* ), ( $( [<T $i>] ),* )> for NamedArgs
            where
                $( NamedArgs: FromNamesArgs<[<Names $i>], [<T $i>]> ),*
            {
                fn from_names_args(names: ( $( [<Names $i>] ),* ), args: ( $( [<T $i>] ),* )) -> Self {
                    let mut out = NamedArgs::new();
                    $(
                        out = out.union(Self::from_names_args(names.$i, args.$i));
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

impl_from_names_args!(0, 1);
impl_from_names_args!(0, 1, 2);
impl_from_names_args!(0, 1, 2, 3);
impl_from_names_args!(0, 1, 2, 3, 4);
impl_from_names_args!(0, 1, 2, 3, 4, 5);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
impl_from_names_args!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

#[cfg(test)]
mod tests {
    use crate::names;

    #[test]
    fn partition_should_return_full_map_for_fst_when_all_args() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert_eq!(
            named_args
                .clone()
                .partition(&names!["foo", "bar"], &names![])
                .unwrap()
                .0
                .into_map::<i32>(),
            named_args.into_map()
        );
    }

    #[test]
    fn partition_should_return_full_map_for_snd_when_all_args() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert_eq!(
            named_args
                .clone()
                .partition(&names![], &names!["foo", "bar"])
                .unwrap()
                .1
                .into_map::<i32>(),
            named_args.into_map()
        );
    }

    #[test]
    fn partition_should_return_maps_with_given_args_when_some_args() {
        let named_args = named_args![("foo", 1), ("bar", 2), ("baz", 3)];
        let (left, right) = named_args
            .partition(&names!["baz"], &names!["foo"])
            .unwrap();
        assert_eq!(left.into_map::<i32>(), named_args![("baz", 3)].into_map(),);
        assert_eq!(right.into_map::<i32>(), named_args![("foo", 1)].into_map(),);
    }

    #[test]
    fn partition_should_duplicate_args_required_by_both() {
        let named_args = named_args![("foo", 1), ("bar", 2), ("baz", 3), ("biz", 4)];
        let (left, right) = named_args
            .partition(&names!["foo", "bar", "baz"], &names!["foo", "bar", "biz"])
            .unwrap();
        assert_eq!(
            left.into_map::<i32>(),
            named_args![("foo", 1), ("bar", 2), ("baz", 3)].into_map(),
        );
        assert_eq!(
            right.into_map::<i32>(),
            named_args![("foo", 1), ("bar", 2), ("biz", 4)].into_map(),
        );
    }

    #[test]
    fn partition_should_return_none_if_arg_missing() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert!(named_args
            .clone()
            .partition(&names![], &names!["foo", "baz"])
            .is_err());
        assert!(named_args
            .partition(&names!["foo", "baz"], &names![])
            .is_err());
    }

    #[test]
    fn partition3_should_return_full_map_for_fst_when_all_args() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert_eq!(
            named_args
                .clone()
                .partition3(&names!["foo", "bar"], &names![], &names![])
                .unwrap()
                .0
                .into_map::<i32>(),
            named_args.into_map()
        );
    }

    #[test]
    fn partition3_should_return_full_map_for_snd_when_all_args() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert_eq!(
            named_args
                .clone()
                .partition3(&names![], &names!["foo", "bar"], &names![])
                .unwrap()
                .1
                .into_map::<i32>(),
            named_args.into_map()
        );
    }

    #[test]
    fn partition3_should_return_full_map_for_third_when_all_args() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert_eq!(
            named_args
                .clone()
                .partition3(&names![], &names![], &names!["foo", "bar"])
                .unwrap()
                .2
                .into_map::<i32>(),
            named_args.into_map()
        );
    }

    #[test]
    fn partition3_should_return_maps_with_given_args_when_some_args() {
        let named_args = named_args![("foo", 1), ("bar", 2), ("baz", 3), ("bin", 4)];
        let (fst, snd, third) = named_args
            .partition3(&names!["baz"], &names!["foo"], &names!["bar"])
            .unwrap();
        assert_eq!(fst.into_map::<i32>(), named_args![("baz", 3)].into_map(),);
        assert_eq!(snd.into_map::<i32>(), named_args![("foo", 1)].into_map(),);
        assert_eq!(third.into_map::<i32>(), named_args![("bar", 2)].into_map(),);
    }

    #[test]
    fn partition3_should_duplicate_args_required_by_more_than_one() {
        let named_args = named_args![("foo", 1), ("bar", 2), ("baz", 3), ("biz", 4), ("bop", 5)];
        let (fst, snd, third) = named_args
            .partition3(
                &names!["foo", "bar", "baz"],
                &names!["foo", "bar", "biz"],
                &names!["bar", "biz", "bop"],
            )
            .unwrap();
        assert_eq!(
            fst.into_map::<i32>(),
            named_args![("foo", 1), ("bar", 2), ("baz", 3)].into_map(),
        );
        assert_eq!(
            snd.into_map::<i32>(),
            named_args![("foo", 1), ("bar", 2), ("biz", 4)].into_map(),
        );
        assert_eq!(
            third.into_map::<i32>(),
            named_args![("bar", 2), ("biz", 4), ("bop", 5)].into_map(),
        );
    }

    #[test]
    fn partition3_should_return_none_if_arg_missing() {
        let named_args = named_args![("foo", 1), ("bar", 2)];
        assert!(named_args
            .clone()
            .partition3(&names!["foo", "baz"], &names![], &names![])
            .is_err());
        assert!(named_args
            .clone()
            .partition3(&names![], &names!["foo", "baz"], &names![])
            .is_err());
        assert!(named_args
            .partition3(&names![], &names![], &names!["foo", "baz"])
            .is_err());
    }

    #[test]
    fn pop_should_return_arg_if_available_and_correct_type() {
        let mut named_args = named_args![("foo", 1), ("bar", 2.0)];
        assert_eq!(named_args.pop::<i32>("foo").ok(), Some(1));
        assert_eq!(named_args.pop::<f64>("bar").ok(), Some(2.0));
    }

    #[test]
    fn pop_should_return_none_if_not_available() {
        let mut named_args = named_args![("foo", 1), ("bar", 2.0)];
        assert_eq!(named_args.pop::<i32>("baz").ok(), None);
    }

    #[test]
    fn pop_should_return_none_if_wrong_type() {
        let mut named_args = named_args![("foo", 1), ("bar", 2.0)];
        assert_eq!(named_args.pop::<f64>("foo").ok(), None);
    }
}
