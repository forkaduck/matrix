use std::fmt::Debug;
use std::ops;

use ocl::{Buffer, Kernel};

use crate::Matrix;

pub mod test;

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
        let buffer_size = self.A.len();

        // Buffers
        let buffer_rhs = Buffer::<T>::builder()
            .len(buffer_size)
            .queue(self.loader.queue.clone())
            .build()
            .expect("buffer rhs");

        let buffer_lhs = Buffer::<T>::builder()
            .len(buffer_size)
            .queue(self.loader.queue.clone())
            .build()
            .expect("buffer lhs");

        let buffer_output = Buffer::<T>::builder()
            .len(buffer_size)
            .queue(self.loader.queue.clone())
            .build()
            .expect("buffer out");

        buffer_rhs.write(&rhs.A).enq().expect("write to rhs");
        buffer_lhs.write(&self.A).enq().expect("write to lhs");

        // Build the kernel
        let kernel = match Kernel::builder()
            .program(&self.loader.program)
            .name(kernel_name)
            .queue(self.loader.queue.clone())
            .global_work_size(1 << 10)
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
        let mut rv = Matrix {
            loader: self.loader,
            A: Vec::with_capacity(self.A.len()),
        };

        // Has to be preallocated
        rv.A.resize(rv.A.capacity(), T::default());

        buffer_output
            .read(&mut rv.A)
            .len(self.A.capacity())
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
