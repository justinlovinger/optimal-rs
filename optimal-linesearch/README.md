[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

This package is experimental.
Expect frequent updates to the repository
with breaking changes
and infrequent releases.

# optimal-linesearch

Line-search optimizers.

Fixed step-size optimization can also be performed
using this package:

```rust
use computation_types::*;
use optimal_linesearch::{descend, step_direction::steepest_descent, StepSize};

println!(
    "{:?}",
    val!(0)
        .zip(val1!(vec![10.0, 10.0]))
        .loop_while(
            ("i", "point"),
            (arg!("i", usize) + val!(1)).zip(descend(
                val!(StepSize::new(0.5).unwrap()),
                steepest_descent(val!(2.0) * arg1!("point", f64)),
                arg1!("point", f64),
            )),
            arg!("i", usize).lt(val!(10)),
        )
        .snd()
        .run()
)
```

See [`backtracking_line_search`] for more sophisticated and effective optimizers.

License: MIT
