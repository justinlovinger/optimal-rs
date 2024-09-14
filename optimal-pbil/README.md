[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

```rust
use optimal_compute_core::{argvals, run::Value, Run};
use optimal_pbil::PbilBuilder;

let pbil = PbilBuilder::default()
    .for_(2, |point| Value(point.iter().filter(|x| **x).count()))
    .computation();
println!("{}", pbil);
println!("{:?}", pbil.run(argvals![]));
```

License: MIT
