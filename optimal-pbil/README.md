[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

```rust
use computation_types::{arg1, named_args, peano::Zero, Computation, Run};
use optimal_pbil::PbilBuilder;

let pbil = PbilBuilder::default()
    .for_(
        2,
        arg1!("sample").black_box::<_, _, usize>(|sample: Vec<bool>| {
            sample.iter().filter(|x| **x).count()
        }),
    )
    .computation();
println!("{}", pbil);
println!("{:?}", pbil.run());
```

License: MIT
