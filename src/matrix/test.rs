#[cfg(test)]
mod matrix_tests {
    use crate::matrix::Matrix;
    use matrix_macro::matrix_new;
    use rand::{prelude::*, thread_rng};

    #[test]
    fn new_instance() {
        let mut test = matrix_new!(f64; 1);

        test.data.push(1.0);
    }

    #[test]
    fn add_2d() {
        let mut one = matrix_new!(f32; 1);
        let mut two = matrix_new!(f32; 1);

        for _ in 0..10 {
            one.data.push(thread_rng().gen());
            two.data.push(thread_rng().gen());
        }

        println!("{:?}\n{:?}", one, two);

        let output = &one + &two;

        println!("{:?}", output);

        let output = &output + &one;

        println!("{:?}", output);
    }
}
