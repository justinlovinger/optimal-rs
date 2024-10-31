use crate::{ComputationFn, FromNamesArgs, NamedArgs};

#[derive(Clone, Copy, Debug)]
pub struct Function<ArgNames, Body> {
    pub name: Option<&'static str>,
    pub arg_names: ArgNames,
    pub body: Body,
}

impl<ArgNames, Body> Function<ArgNames, Body> {
    pub fn anonymous(arg_names: ArgNames, body: Body) -> Self {
        Self {
            name: None,
            arg_names,
            body,
        }
    }

    pub fn named(name: &'static str, arg_names: ArgNames, body: Body) -> Self {
        Self {
            name: Some(name),
            arg_names,
            body,
        }
    }

    pub fn fill<Args>(self, args: Args) -> Body::Filled
    where
        NamedArgs: FromNamesArgs<ArgNames, Args>,
        Body: ComputationFn,
    {
        self.body
            .fill(NamedArgs::from_names_args(self.arg_names, args))
    }
}
