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

// Helps to implement all basic operations.
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

// Helps with the *Assign traits.
macro_rules! assign_oper_impl {
    ($op: ident, $opfn: ident, $kernel: ident) => {
        impl<'a, T> ops::$op<&'a Matrix<'_, Vec<T>>> for Matrix<'a, Vec<T>>
        where
            T: ocl::OclPrm,
        {
            fn $opfn(&mut self, rhs: &'a Matrix<'_, Vec<T>>) {
                *self = self.basic_op(rhs, std::stringify!($kernel));
            }
        }
    };
}

assign_oper_impl!(AddAssign, add_assign, add);
assign_oper_impl!(SubAssign, sub_assign, sub);
assign_oper_impl!(MulAssign, mul_assign, mul);
assign_oper_impl!(DivAssign, div_assign, div);

// impl<'a, T> ops::AddAssign<&'a Matrix<'_, Vec<T>>> for T
// where
// T: ocl::OclPrm,
// {
// fn add_assign(&mut self, rhs: &'a Matrix<'_, Vec<T>>) {
// *self = rhs.vec_op(rhs, "add");
// }
// }
