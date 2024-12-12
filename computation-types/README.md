[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# computation-types

Types for abstract mathematical computation.

Note,
documentation is currently lacking.
The best way to learn about this framework
is to read the tests
and see how it is used to implement algorithms
in Optimal.

## Examples

```rust
use computation_types::{named_args, val, Run};

let one_plus_one = val!(1) + val!(1);
assert_eq!(one_plus_one.to_string(), "(1 + 1)");
assert_eq!(one_plus_one.run(), 2);
```

License: MIT
