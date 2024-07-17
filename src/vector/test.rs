#[cfg(test)]
mod matrix_tests {
    use half::f16;
    use log::{info, warn};
    use oorandom;
    use simplelog::{ColorChoice, Config, LevelFilter, TermLogger, TerminalMode};
    use std::ops::*;
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::loader::KernelLoader;
    use crate::Matrix;
    use matrix_macro::matrix_new;

    pub fn timer_end(start: Instant) {
        println!("Time elapsed: {}s", (Instant::now() - start).as_secs_f64());
    }

    pub fn setup() {
        match TermLogger::init(
            LevelFilter::Debug,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ) {
            Ok(_) => {}
            Err(_) => {}
        };

        println!("");
    }

    fn vec_ops<T, const VAL_LEN: usize>()
    where
        T: Add<Output = T>
            + AddAssign
            + Sub<Output = T>
            + Mul<Output = T>
            + MulAssign
            + Div<Output = T>
            + ocl::OclPrm
            + std::convert::From<u8>
            + std::convert::Into<f64>,
    {
        setup();
        let start = Instant::now();

        let loader = KernelLoader::new::<T>(&PathBuf::from("./kernels"), false, false, 16).unwrap();

        let mut one = matrix_new!(&loader, T, 1, VAL_LEN);
        let mut two = matrix_new!(&loader, T, 1, VAL_LEN);

        let mut rng = oorandom::Rand32::new(10);

        for _ in 0..VAL_LEN {
            let temp = (rng.rand_u32() as u8) % 10;
            one.A.push(temp.into());

            let temp = (rng.rand_u32() as u8) % 10;
            two.A.push(temp.into());
        }

        macro_rules! normal_op_test {
            ($op: ident, $name: ident) => {
                let result = &one.$op(&two);
                info!("{}:\t{:?}", std::stringify!($name), result);

                for i in 0..result.A.len() {
                    assert_eq!(one.A[i].$op(two.A[i]), result.A[i]);
                }
            };
        }

        info!("Input:");
        info!("1: \t{:?}", one);
        info!("2: \t{:?}", two);

        // Check +
        normal_op_test!(add, Add);

        let mut result = Matrix {
            loader: &loader,
            A: T::default(),
        };
        result += &one;

        let mut sec_result = one.A[0];
        for i in 1..one.A.len() {
            sec_result += one.A[i];
        }
        assert_eq!(result.A, sec_result);

        // Check -
        normal_op_test!(sub, Sub);

        // Check *
        normal_op_test!(mul, Mul);

        let mut result = Matrix {
            loader: &loader,
            A: T::default(),
        };
        result *= &one;

        let mut sec_result = one.A[0];
        for i in 1..one.A.len() {
            sec_result *= one.A[i];
        }
        assert_eq!(result.A, sec_result);

        // Check /
        let result = &one / &two;
        info!("Div:\t{:?}", result);

        for i in 0..result.A.len() {
            let quotient: f64 = (one.A[i] / two.A[i]).into();
            let out = result.A[i].into();

            if quotient != out {
                warn!(
                    "[{}] Rounding mode differs!\nCPU: {:b}\nGPU: {:b}",
                    i,
                    quotient.to_bits(),
                    out.to_bits()
                );
            }
        }

        timer_end(start);
    }

    // Test all possible matrix types.
    #[test]
    fn vec_ops_f16() {
        vec_ops::<f16, 10>();
    }

    #[test]
    fn vec_ops_f32() {
        vec_ops::<f32, 10>();
    }

    #[test]
    fn vec_ops_f64() {
        vec_ops::<f64, 10>();
    }

    // Check for some nice index, or off-by-one errors.
    #[test]
    fn vec_ops_f32_large() {
        vec_ops::<f32, 100>();
    }
}
