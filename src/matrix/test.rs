#[cfg(test)]
mod matrix_tests {
    use crate::loader::KernelLoader;
    use crate::matrix::Matrix;
    use matrix_macro::matrix_new;
    use rand::{prelude::*, thread_rng};
    use simple_logger::SimpleLogger;
    use std::path::PathBuf;

    #[test]
    fn vec_ops() {
        SimpleLogger::new().init().unwrap();

        let mut loader = KernelLoader::new::<f32>(&PathBuf::from("./kernels")).unwrap();
        loader.proque.set_dims(1 << 10);

        let mut one = matrix_new!(&loader, f32, 1);
        let mut two = matrix_new!(&loader, f32, 1);

        for _ in 0..10 {
            one.data.push(thread_rng().gen());
            two.data.push(thread_rng().gen());
        }

        println!("{}\n{}", one, two);

        let output = &one + &two;

        println!("{}", output);

        let output = &one - &two;

        println!("{}", output);
    }
}
