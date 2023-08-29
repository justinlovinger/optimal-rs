# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

```rust
use ndarray::prelude::*;
use optimal_pbil::*;

println!(
    "{:?}",
    UntilProbabilitiesConvergedConfig::default()
        .start(Config::start_default_for(16, |point| point.iter().filter(|x| **x).count()))
        .argmin()
);
```

License: MIT
