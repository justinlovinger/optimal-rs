[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

```rust
use optimal_compute_core::{arg1, argvals, peano::Zero, run::Value, Computation, Run};
use optimal_pbil::PbilBuilder;

let pbil = PbilBuilder::default()
    .for_(
        2,
        arg1!("sample").black_box::<_, Zero, usize>(|sample: Vec<bool>| {
            Value(sample.iter().filter(|x| **x).count())
        }),
    )
    .computation();
println!("{}", pbil);
println!("{:?}", pbil.run(argvals![]));
```

License: MIT
