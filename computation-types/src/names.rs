use std::collections::BTreeSet;

pub type Name = &'static str;

#[derive(Clone, Debug)]
pub struct Names(BTreeSet<Name>);

impl Default for Names {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! names {
    ( ) => {
        $crate::Names::new()
    };
    ( $name:literal ) => {
        $crate::Names::singleton($name)
    };
    ( $name:literal, $( $rest:tt ),* ) => {
        $crate::Names::singleton($name).union(names![$( $rest ),*])
    };
}

impl Names {
    pub fn new() -> Self {
        Names(BTreeSet::new())
    }

    pub fn singleton(name: Name) -> Self {
        Names(std::iter::once(name).collect())
    }

    pub fn union_many<'a>(names: impl IntoIterator<Item = &'a Names>) -> Self {
        let mut set = BTreeSet::new();
        for name in names.into_iter().flat_map(|names| names.iter()) {
            set.insert(name);
        }
        Names(set)
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
