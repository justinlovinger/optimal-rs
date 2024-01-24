[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal

Mathematical optimization and machine learning framework
and algorithms.

Optimal provides a composable framework
for mathematical optimization
and machine learning
from the optimization perspective,
in addition to algorithm implementations.

The framework consists of runners,
optimizers,
and problems,
with a chain of dependency as follows:
`runner -> optimizer -> problem`.
Most optimizers can support many problems
and most runners can support many optimizers.

A problem defines a mathematical optimization problem.
An optimizer defines the steps for solving a problem,
usually as an infinite series of state transitions
incrementally improving a solution.
A runner defines the stopping criteria for an optimizer
and may affect the optimization sequence
in other ways.

## Examples

Minimize the "count" problem
using a derivative-free optimizer:

```rust
use optimal::binary_argmin;

println!(
    "{:?}",
    binary_argmin(
        0,
        2,
        |point| point.iter().filter(|x| **x).count() as f64,
    )
);
```

Minimize the "sphere" problem
using a derivative optimizer:

```rust
use optimal::real_derivative_argmin;

println!(
    "{:?}",
    real_derivative_argmin(
        0,
        std::iter::repeat(-10.0..=10.0).take(2),
        |point| point.iter().map(|x| x.powi(2)).sum(),
        |point| point.iter().map(|x| 2.0 * x).collect(),
    )
);
```

For more control over configuration parameters,
introspection of the optimization process,
serialization,
and specialization that may improve performance,
see individual optimizer packages.

License: MIT
