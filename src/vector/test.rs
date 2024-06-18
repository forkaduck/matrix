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

    pub fn setup() -> Instant {
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
        Instant::now()
    }

    fn vec_ops<T>()
    where
        T: Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + ocl::OclPrm
            + std::convert::From<u8>
            + std::convert::Into<f64>,
    {
        let start = setup();

        let loader = KernelLoader::new::<T>(&PathBuf::from("./kernels"), false).unwrap();

        const VAL_LEN: usize = 5;

        let mut one = matrix_new!(&loader, T, 1, VAL_LEN);
        let mut two = matrix_new!(&loader, T, 1, VAL_LEN);

        let mut rng = oorandom::Rand64::new(10);

        for _ in 0..VAL_LEN {
            let temp = rng.rand_u64() as u8;
            one.A.push(temp.into());

            let temp = rng.rand_u64() as u8;
            two.A.push(temp.into());
        }

        info!("Input:");
        info!("1: \t{:?}", one);
        info!("2: \t{:?}", two);

        let output = &one + &two;
        info!("Add:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(one.A[i] + two.A[i], output.A[i]);
        }

        let output = &one - &two;
        info!("Sub:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(one.A[i] - two.A[i], output.A[i]);
        }

        let output = &one * &two;
        info!("Mult:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(one.A[i] * two.A[i], output.A[i]);
        }

        let output = &one / &two;
        info!("Div:\t{:?}", output);

        for i in 0..output.A.len() {
            let quotient: f64 = (one.A[i] / two.A[i]).into();
            let out = output.A[i].into();

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

    #[test]
    fn vec_ops_f16() {
        vec_ops::<f16>();
    }

    #[test]
    fn vec_ops_f32() {
        vec_ops::<f32>();
    }

    #[test]
    fn vec_ops_f64() {
        vec_ops::<f64>();
    }
}
