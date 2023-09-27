[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

```rust
use optimal_pbil::*;

println!(
    "{:?}",
    UntilProbabilitiesConvergedConfig::default()
        .start(Config::start_default_for(16, |point| point.iter().filter(|x| **x).count()))
        .argmin()
);
```

License: MIT
