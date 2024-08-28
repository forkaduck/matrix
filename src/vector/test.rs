#[cfg(test)]
mod matrix_tests {
    use half::f16;
    use log::{info, warn};
    use oorandom;
    use simplelog::{ColorChoice, Config, LevelFilter, TermLogger, TerminalMode};
    use std::ops::*;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    use crate::loader::KernelLoader;
    use crate::Matrix;
    use matrix_macro::matrix_new;

    const TXTSHIFT: &str = "\x1b[100G";

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

        let loader = Arc::new(
            KernelLoader::new::<T>(&PathBuf::from("./kernels"), false, false, 16).unwrap(),
        );

        let mut lhs = matrix_new!(loader.clone(), T, 1, VAL_LEN);
        let mut rhs = matrix_new!(loader.clone(), T, 1, VAL_LEN);

        let rhs_scalar: T = 10u8.into();

        let mut result = matrix_new!(loader.clone(), T, 1, VAL_LEN);
        let mut result_scalar = Matrix {
            loader: Some(loader.clone()),
            A: T::default(),
        };

        let mut rng = oorandom::Rand32::new(10);

        for _ in 0..VAL_LEN {
            let temp = (rng.rand_u32() as u8) % 10;
            lhs.A.push(temp.into());

            let temp = (rng.rand_u32() as u8) % 10;
            rhs.A.push(temp.into());
        }

        macro_rules! normal_op_test {
            ($op: ident, $name: literal, $rhs: expr, $chill: expr) => {
                result = lhs.$op(&rhs);
                info!("{:?}{}:{}", result, TXTSHIFT, $name);

                for i in 0..result.A.len() {
                    if $chill == true {
                        let cp_out: f64 = (lhs.A[i].$op(rhs.A[i])).into();
                        let gp_out = result.A[i].into();

                        if cp_out != gp_out {
                            warn!(
                                "[{}] Rounding mode differs!\nCPU: {:b}\nGPU: {:b}",
                                i,
                                cp_out.to_bits(),
                                gp_out.to_bits()
                            );
                        }
                    } else {
                        assert_eq!(lhs.A[i].$op(rhs.A[i]), result.A[i]);
                    }
                }
            };
        }

        info!("Input:");
        info!("{:?}{}:lhs", lhs, TXTSHIFT);
        info!("{:?}{}:rhs", rhs, TXTSHIFT);

        normal_op_test!(add, "Matrix<Vec<T>> + Matrix<Vec<T>>", rhs, false);
        normal_op_test!(add, "Matrix<Vec<T>> + [T]", rhs.A[..], false);

        result_scalar += &lhs;

        let mut temp = lhs.A[0];
        for i in 1..lhs.A.len() {
            temp += lhs.A[i];
        }
        info!("{:?}{}:Matrix<T> += Matrix<Vec<T>>", result, TXTSHIFT);
        assert_eq!(result_scalar.A, temp);

        result = &lhs + rhs_scalar;
        info!(
            "{:?}{}:Matrix<Vec<T>> = Matrix<Vec<T>> + T",
            result, TXTSHIFT
        );
        for i in 0..lhs.A.len() {
            assert_eq!(result.A[i], lhs.A[i] + rhs_scalar);
        }

        normal_op_test!(sub, "Matrix<Vec<T>> - Matrix<Vec<T>>", rhs, false);
        normal_op_test!(sub, "Matrix<Vec<T>> - [T]", rhs.A[..], false);

        result = &lhs - rhs_scalar;
        info!(
            "{:?}{}:Matrix<Vec<T>> = Matrix<Vec<T>> - T",
            result, TXTSHIFT
        );
        for i in 0..lhs.A.len() {
            assert_eq!(result.A[i], lhs.A[i] - rhs_scalar);
        }

        normal_op_test!(mul, "Matrix<Vec<T>> * Matrix<Vec<T>>", rhs, false);
        normal_op_test!(mul, "Matrix<Vec<T>> * [T]", rhs.A[..], false);

        result = &lhs * rhs_scalar;
        info!(
            "{:?}{}:Matrix<Vec<T>> = Matrix<Vec<T>> * T",
            result, TXTSHIFT
        );
        for i in 0..lhs.A.len() {
            assert_eq!(result.A[i], lhs.A[i] * rhs_scalar);
        }

        result_scalar *= &lhs;

        let mut temp = lhs.A[0];
        for i in 1..lhs.A.len() {
            temp *= lhs.A[i];
        }
        info!("{:?}{}:Matrix<T> *= Matrix<Vec<T>>", result, TXTSHIFT);
        assert_eq!(result_scalar.A, temp);

        normal_op_test!(div, "Matrix<Vec<T>> / Matrix<Vec<T>>", rhs, true);
        normal_op_test!(div, "Matrix<Vec<T>> / [T]", rhs.A[..], true);

        result = &lhs / rhs_scalar;
        info!(
            "{:?}{}:Matrix<Vec<T>> = Matrix<Vec<T>> / T",
            result, TXTSHIFT
        );
        for i in 0..lhs.A.len() {
            let quotient: f64 = (lhs.A[i] / rhs_scalar).into();
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
