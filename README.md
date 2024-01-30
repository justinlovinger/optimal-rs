[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal

Mathematical optimization and machine-learning components and algorithms.

Optimal provides composable functions
for mathematical optimization
and machine-learning
from the optimization-perspective.

## Examples

Minimize the "count" problem
using a derivative-free optimizer:

```rust
use optimal::Binary;

println!(
    "{:?}",
    Binary::default()
        .for_(2, |point| point.iter().filter(|x| **x).count() as f64)
        .argmin()
);
```

Minimize the "sphere" problem
using a derivative optimizer:

```rust
use optimal::RealDerivative;

println!(
    "{:?}",
    RealDerivative::default()
        .for_(
            std::iter::repeat(-10.0..=10.0).take(2),
            |point| point.iter().map(|x| x.powi(2)).sum(),
            |point| point.iter().map(|x| 2.0 * x).collect()
        )
        .argmin()
);
```

For more control over configuration,
introspection of the optimization process,
serialization,
and specialization that may improve performance,
see individual optimizer packages.

Note: more specific support for machine-learning will be added in the future.
Currently,
machine-learning is supported
by defining an objective-function
taking model-parameters.

License: MIT
