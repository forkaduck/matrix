use matrix_macro::matrix_new;
use ocl::ProQue;
use std::ops;

pub mod test;

#[derive(Debug)]
struct Matrix<T> {
    pub data: T,
}

impl<T: ocl::OclPrm> ops::Add<Matrix<Vec<T>>> for Matrix<Vec<T>> {
    type Output = Matrix<Vec<T>>;

    fn add(self, rhs: Matrix<Vec<T>>) -> Matrix<Vec<T>> {
        if self.data.len() != rhs.data.len() {
            panic!("Both operators have to have the same size");
        }

        // Setup
        let src = std::fs::read_to_string("src/matrix/matrix.cl").expect("Failed to read file");
        let proque = match ProQue::builder().src(src).dims(1 << 20).build() {
            Ok(a) => a,
            Err(e) => {
                println!("OpenCL Kernel Compile Error: {}", e);
                panic!();
            }
        };

        // Buffers
        let buffer_rhs = proque.create_buffer::<T>().unwrap();
        let buffer_lhs = proque.create_buffer::<T>().unwrap();
        let buffer_output = proque.create_buffer::<T>().unwrap();

        buffer_rhs.write(&rhs.data).enq().unwrap();
        buffer_lhs.write(&self.data).enq().unwrap();

        // Build the kernel
        let kernel = match proque
            .kernel_builder("add")
            .arg(&buffer_rhs)
            .arg(buffer_rhs.len() as u32)
            .arg(&buffer_lhs)
            .arg(buffer_lhs.len() as u32)
            .arg(&buffer_output)
            .build()
        {
            Ok(a) => a,
            Err(e) => {
                println!("OpenCL Build Error: {}", e);
                panic!();
            }
        };

        unsafe {
            kernel.enq().unwrap();
        }

        // Get results
        let mut rv = matrix_new!(T; 1);

        for _ in 0..self.data.len() {
            rv.data.push(T::default());
        }

        buffer_output.read(&mut rv.data).enq().unwrap();

        rv
    }
}
