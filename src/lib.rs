#![feature(let_chains)]
//#![feature(f16)]

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Matrix<'a, T> {
    pub loader: &'a crate::loader::KernelLoader,
    pub A: T,
}

pub mod loader;
pub mod vector;
pub use matrix_macro::matrix_new;
