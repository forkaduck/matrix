# Matrix
A simple and incomplete linear algebra library which uses OpenCL on a GPU for all calculations.

## Build and run tests
Reducing the amount of threads helps to make the output more readable.
```
$ RUST_LOG=debug cargo test -- --nocapture --test-threads 1
```

__Note:__
Kernels for the f16 and f64 types might not work because the GPU driver doesn't support them yet.
(Looking at you Mesa...)
