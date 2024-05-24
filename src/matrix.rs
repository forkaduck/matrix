use std::fmt::Display;
use std::ops;

pub mod test;

#[derive(Clone)]
pub struct Matrix<'a, T> {
    pub data: T,
    pub loader: &'a crate::loader::KernelLoader,
}

impl<T> Display for Matrix<'_, Vec<T>>
where
    T: std::fmt::Display + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut temp = Vec::new();
        for i in &self.data {
            temp.push(i);
        }
        write!(f, "{:?}", temp)
    }
}

impl<'r, 'l, T> Matrix<'r, Vec<T>>
where
    T: ocl::OclPrm,
    'r: 'l,
{
    fn vec_op(&self, rhs: &'r Matrix<Vec<T>>, kernel_name: &str) -> Matrix<'l, Vec<T>> {
        if self.data.len() != rhs.data.len() {
            panic!("Both operators have to have the same size");
        }

        // Buffers
        let buffer_rhs = self.loader.proque.create_buffer::<T>().expect("buffer rhs");
        let buffer_lhs = self.loader.proque.create_buffer::<T>().expect("buffer lhs");
        let buffer_output = self.loader.proque.create_buffer::<T>().expect("buffer out");

        buffer_rhs.write(&rhs.data).enq().expect("write to rhs");
        buffer_lhs.write(&self.data).enq().expect("write to lhs");

        // Build the kernel
        let kernel = match self
            .loader
            .proque
            .kernel_builder(kernel_name)
            .arg(&buffer_rhs)
            .arg(buffer_rhs.len() as u64)
            .arg(&buffer_lhs)
            .arg(buffer_lhs.len() as u64)
            .arg(&buffer_output)
            .build()
        {
            Ok(a) => a,
            Err(e) => {
                panic!("{}", e);
            }
        };

        unsafe {
            kernel.enq().expect("kernel enque");
        }

        // Get results
        let mut rv = matrix_macro::matrix_new!(self.loader, T, 1);

        rv.data.resize(self.data.len(), T::default());

        buffer_output
            .read(&mut rv.data)
            .len(self.data.len())
            .enq()
            .expect("read from out");
        rv
    }
}

impl<'r, 'l, T> ops::Add<&'r Matrix<'_, Vec<T>>> for &'l Matrix<'_, Vec<T>>
where
    T: ocl::OclPrm,
    'r: 'l,
{
    type Output = Matrix<'l, Vec<T>>;

    fn add(self, rhs: &'r Matrix<Vec<T>>) -> Self::Output {
        self.vec_op(rhs, "add")
    }
}

impl<'r, 'l, T> ops::Sub<&'r Matrix<'_, Vec<T>>> for &'l Matrix<'_, Vec<T>>
where
    T: ocl::OclPrm,
    'r: 'l,
{
    type Output = Matrix<'l, Vec<T>>;

    fn sub(self, rhs: &'r Matrix<Vec<T>>) -> Self::Output {
        self.vec_op(rhs, "sub")
    }
}
