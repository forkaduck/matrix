use std::fmt::Debug;
use std::ops;

pub mod test;

#[allow(non_snake_case)]
#[derive(Clone)]
pub struct Matrix<'a, T> {
    pub loader: &'a crate::loader::KernelLoader,
    pub A: T,
}

impl<T> Debug for Matrix<'_, Vec<T>>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut temp = Vec::new();
        for i in &self.A {
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
        if self.A.len() != rhs.A.len() {
            panic!("Both operators have to have the same size");
        }

        // Buffers
        let buffer_rhs = self.loader.proque.create_buffer::<T>().expect("buffer rhs");
        let buffer_lhs = self.loader.proque.create_buffer::<T>().expect("buffer lhs");
        let buffer_output = self.loader.proque.create_buffer::<T>().expect("buffer out");

        buffer_rhs.write(&rhs.A).enq().expect("write to rhs");
        buffer_lhs.write(&self.A).enq().expect("write to lhs");

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

        rv.A.resize(self.A.len(), T::default());

        buffer_output
            .read(&mut rv.A)
            .len(self.A.len())
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

impl<'r, 'l, T> ops::Mul<&'r Matrix<'_, Vec<T>>> for &'l Matrix<'_, Vec<T>>
where
    T: ocl::OclPrm,
    'r: 'l,
{
    type Output = Matrix<'l, Vec<T>>;

    fn mul(self, rhs: &'r Matrix<Vec<T>>) -> Self::Output {
        self.vec_op(rhs, "mul")
    }
}

impl<'r, 'l, T> ops::Div<&'r Matrix<'_, Vec<T>>> for &'l Matrix<'_, Vec<T>>
where
    T: ocl::OclPrm,
    'r: 'l,
{
    type Output = Matrix<'l, Vec<T>>;

    fn div(self, rhs: &'r Matrix<Vec<T>>) -> Self::Output {
        self.vec_op(rhs, "div")
    }
}
