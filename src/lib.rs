#![feature(let_chains)]
//#![feature(f16)]

pub mod loader;
pub mod vector;
pub use matrix_macro::matrix_new;

use std::sync::Arc;

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Matrix<T> {
    pub loader: Option<Arc<crate::loader::KernelLoader>>,
    pub A: T,
}

impl<T> Matrix<T>
where
    T: std::clone::Clone,
{
    fn fill_dim(&self, size: usize) -> Matrix<Vec<T>> {
        Matrix {
            loader: self.loader.clone(),
            A: vec![self.A.clone(); size],
        }
    }
}
