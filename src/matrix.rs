use matrix_macro::matrix_new;
use std::ops;

pub mod test;

struct Matrix<T> {
    pub data: T,
}

impl<T> ops::Add<Matrix<Vec<Vec<T>>>> for Matrix<Vec<Vec<T>>> {
    type Output = Matrix<Vec<Vec<T>>>;

    fn add(self, rhs: Matrix<Vec<Vec<T>>>) -> Matrix<Vec<Vec<T>>> {
        let temp = matrix_new!(T; 2);

        temp
    }
}
