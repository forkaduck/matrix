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
    fn basic_op(&self, rhs: &'r Matrix<Vec<T>>, kernel_name: &str) -> Matrix<'l, Vec<T>> {
        // Check for common invocation errors.
        debug_assert!(
            self.A.len() == rhs.A.len(),
            "Both operators have to have the same size! lhs:{} != rhs:{}",
            self.A.len(),
            rhs.A.len()
        );
        debug_assert!(self.A.len() != 0, "LHS is empty");
        debug_assert!(rhs.A.len() != 0, "RHS is empty");

        let buffer_size = self.A.len();

        // Create all operator buffers.
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

        // Write the Vec contents to their respective buffer.
        buffer_rhs.write(&rhs.A).enq().expect("write to rhs");
        buffer_lhs.write(&self.A).enq().expect("write to lhs");

        // Run the kernel.
        let kernel = match Kernel::builder()
            .program(&self.loader.program)
            .name(kernel_name)
            .queue(self.loader.queue.clone())
            .global_work_size(self.loader.global_work_size)
            .local_work_size(self.loader.local_work_size)
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

        // Package the results.
        let mut result = Matrix {
            loader: self.loader,
            A: vec![T::default(); buffer_size],
        };

        buffer_output
            .read(&mut result.A)
            .len(buffer_size)
            .enq()
            .expect("read from out");

        result
    }

    fn down_op(&self, kernel_name: &str) -> Matrix<'_, T> {
        // Check for common invocation errors.
        debug_assert!(self.A.len() != 0, "RHS is empty");

        // Create buffers and initialize them.
        let buffer_rhs = Buffer::<T>::builder()
            .len(self.A.len())
            .queue(self.loader.queue.clone())
            .build()
            .expect("buffer rhs");

        buffer_rhs.write(&self.A).enq().expect("write to rhs");

        // Build and run the kernel.
        let kernel = match Kernel::builder()
            .program(&self.loader.program)
            .name(kernel_name)
            .queue(self.loader.queue.clone())
            .global_work_size(self.loader.global_work_size)
            .local_work_size(self.loader.local_work_size)
            .arg(&buffer_rhs)
            .arg(buffer_rhs.len() as u64)
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

        // Read the output from device memory.
        let mut result: Vec<T> = vec![T::default(); 1];

        buffer_rhs
            .read(&mut result)
            .len(1)
            .enq()
            .expect("read from out");

        Matrix {
            loader: &self.loader,
            A: result[0],
        }
    }
}

// Implementation of Matrix<Vec<T>> = Matrix<Vec<T>> @ Matrix<Vec<T>>
macro_rules! oper_impl {
    ($op: ident, $kernel: ident) => {
        impl<'a, T> ops::$op<&'a Matrix<'_, Vec<T>>> for &'a Matrix<'_, Vec<T>>
        where
            T: ocl::OclPrm,
        {
            type Output = Matrix<'a, Vec<T>>;

            fn $kernel(self, rhs: &'a Matrix<Vec<T>>) -> Self::Output {
                self.basic_op(rhs, std::stringify!($kernel))
            }
        }
    };
}

oper_impl!(Add, add);
oper_impl!(Sub, sub);
oper_impl!(Mul, mul);
oper_impl!(Div, div);

// Implementation of Matrix<T> @= Matrix<Vec<T>>
macro_rules! assign_down_impl {
    ($op: ident, $opfn: ident, $kernel: ident) => {
        impl<'a, T> ops::$op<&'a Matrix<'_, Vec<T>>> for Matrix<'a, T>
        where
            T: ocl::OclPrm,
        {
            fn $opfn(&mut self, rhs: &'a Matrix<'_, Vec<T>>) {
                *self = rhs.down_op(std::stringify!($kernel));
            }
        }
    };
}

assign_down_impl!(AddAssign, add_assign, add_down);
assign_down_impl!(MulAssign, mul_assign, mul_down);
