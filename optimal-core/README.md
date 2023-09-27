[![Workflow Status](https://github.com/justinlovinger/optimal-rs/workflows/build/badge.svg)](https://github.com/justinlovinger/optimal-rs/actions?query=workflow%3A%22build%22)

# optimal-core

Core traits and types for Optimal.

Most optimizers are expected to adhere to particular conventions.
An optimizer configuration should remain static during operation.
A problem and configuration
can be used to start a running optimizer.
A running optimizer has state
that depends on its problem
and configuration.

License: MIT
