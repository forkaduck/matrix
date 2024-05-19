#[cfg(test)]
mod matrix_tests {
    use crate::matrix::Matrix;
    use matrix_macro::matrix_new;

    #[test]
    fn basic() {
        let mut test = matrix_new!(f64; 1);

        test.data.push(1.0);
    }
}
