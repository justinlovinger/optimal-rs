use std::collections::BTreeSet;

pub type Name = &'static str;

#[derive(Clone, Debug)]
pub struct Args(BTreeSet<Name>);

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! args {
    ( ) => {
        $crate::Args::new()
    };
    ( $name:literal ) => {
        $crate::Args::singleton($name)
    };
    ( $name:literal, $( $rest:tt ),* ) => {
        $crate::Args::singleton($name).union(args![$( $rest ),*])
    };
}

impl Args {
    pub fn new() -> Self {
        Args(BTreeSet::new())
    }

    pub fn singleton(name: Name) -> Self {
        Args(std::iter::once(name).collect())
    }

    pub fn from_args<'a>(args: impl IntoIterator<Item = &'a Args>) -> Self {
        let mut set = BTreeSet::new();
        for arg in args.into_iter().flat_map(|args| args.iter()) {
            set.insert(arg);
        }
        Args(set)
    }

    pub fn union(mut self, mut other: Self) -> Self {
        self.0.append(&mut other.0);
        self
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, name: Name) -> bool {
        self.0.contains(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = Name> + '_ {
        self.0.iter().copied()
    }
}
