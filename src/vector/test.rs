#[cfg(test)]
mod matrix_tests {
    use log::info;
    use oorandom;
    use simplelog::{ColorChoice, Config, LevelFilter, TermLogger, TerminalMode};
    use std::path::PathBuf;

    use crate::loader::KernelLoader;
    use crate::Matrix;
    use matrix_macro::matrix_new;

    #[test]
    fn vec_ops() {
        TermLogger::init(
            LevelFilter::Debug,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )
        .unwrap();

        let mut loader = KernelLoader::new::<f16>(&PathBuf::from("./kernels")).unwrap();
        loader.proque.set_dims(1 << 15);

        let mut one = matrix_new!(&loader, f16, 1, 10);
        let mut two = matrix_new!(&loader, f16, 1, 10);

        let mut rng = oorandom::Rand64::new(10);

        for _ in 0..10 {
            one.A.push(rng.rand_float() as f16);
            two.A.push(rng.rand_float() as f16);
        }

        info!("Input:");
        info!("1: \t{:?}", one);
        info!("2: \t{:?}", two);

        let output = &one + &two;
        info!("Add:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(output.A[i] == one.A[i] + two.A[i], true);
        }

        let output = &one - &two;
        info!("Sub:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(output.A[i] == one.A[i] - two.A[i], true);
        }

        let output = &one * &two;
        info!("Mult:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(output.A[i] == one.A[i] * two.A[i], true);
        }

        let output = &one / &two;
        info!("Div:\t{:?}", output);

        for i in 0..output.A.len() {
            assert_eq!(output.A[i] == one.A[i] / two.A[i], true);
        }
    }
}
