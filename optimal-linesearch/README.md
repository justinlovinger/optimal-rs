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
use optimal_linesearch::{descend, step_direction::steepest_descent, StepSize};

fn main() {
    let step_size = StepSize::new(0.5).unwrap();
    let mut point = vec![10.0, 10.0];
    for _ in 0..10 {
        point = descend(step_size, steepest_descent(obj_func_d(&point)), point).collect();
    }
    println!("{:?}", point);
}

fn obj_func_d(point: &[f64]) -> Vec<f64> {
    point.iter().copied().map(|x| 2.0 * x).collect()
}
```

See [`backtracking_line_search`] for more sophisticated and effective optimizers.

License: MIT
