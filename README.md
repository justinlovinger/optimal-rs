[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

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

Minimize the `Count` problem
using a PBIL optimizer:

```rust
use ndarray::prelude::*;
use optimal_pbil::*;
use optimal::prelude::*;

println!(
    "{}",
    UntilConvergedConfig::default().argmin(&mut Config::start_default_for(16, |points| {
        points.map_axis(Axis(1), |bits| bits.iter().filter(|x| **x).count())
    }))
);
```

Minimize a problem
one step at a time:

```rust
use ndarray::prelude::*;
use optimal_pbil::*;
use optimal::prelude::*;

let mut it = UntilConvergedConfig::default().start(Config::start_default_for(16, |points| {
    points.map_axis(Axis(1), |bits| bits.iter().filter(|x| **x).count())
}));
while let Some(o) = it.next() {
    println!("{:?}", o.state());
}
let o = it.into_inner().0;
println!("f({}) = {}", o.best_point(), o.best_point_value());
```

License: MIT
