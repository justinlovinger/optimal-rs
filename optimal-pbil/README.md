# optimal-pbil

Population-based incremental learning (PBIL).

## Examples

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

License: MIT
