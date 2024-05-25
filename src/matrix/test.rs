#[cfg(test)]
mod matrix_tests {
    use crate::loader::KernelLoader;
    use crate::matrix::Matrix;
    use log::info;
    use matrix_macro::matrix_new;
    use rand::{prelude::*, thread_rng};
    use simple_logger::SimpleLogger;
    use std::path::PathBuf;

    #[test]
    fn vec_ops() {
        SimpleLogger::new().init().unwrap();

        let mut loader = KernelLoader::new::<f32>(&PathBuf::from("./kernels")).unwrap();
        loader.proque.set_dims(1 << 15);

        let mut one = matrix_new!(&loader, f32, 1);
        let mut two = matrix_new!(&loader, f32, 1);

        for _ in 0..10 {
            one.data.push(thread_rng().gen());
            two.data.push(thread_rng().gen());
        }

        info!("Input:");
        info!("1: \t\t{}", one);
        info!("2: \t\t{}", two);

        let output = &one + &two;
        info!("Add:\t{}", output);

        for i in 0..output.data.len() {
            assert_eq!(output.data[i] == one.data[i] + two.data[i], true);
        }

        let output = &one - &two;
        info!("Sub:\t{}", output);

        for i in 0..output.data.len() {
            assert_eq!(output.data[i] == one.data[i] - two.data[i], true);
        }

        let output = &one * &two;
        info!("Mult:\t{}", output);

        for i in 0..output.data.len() {
            assert_eq!(output.data[i] == one.data[i] * two.data[i], true);
        }

        let output = &one / &two;
        info!("Div:\t{}", output);

        for i in 0..output.data.len() {
            assert_eq!(output.data[i] == one.data[i] / two.data[i], true);
        }
    }
}
