use matrix_macro::matrix_new;
use std::ops;

pub mod test;

#[derive(Debug, Clone)]
struct Matrix<T> {
    pub data: T,
}

impl<'a, 'b, T: ocl::OclPrm> ops::Add<&'b Matrix<Vec<T>>> for &'a Matrix<Vec<T>> {
    type Output = Matrix<Vec<T>>;

    fn add(self, rhs: &Matrix<Vec<T>>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!("Both operators have to have the same size");
        }

        use crate::loader::KernelLoader;
        use std::path::PathBuf;

        let mut loader = KernelLoader::new(&PathBuf::from("./kernels"));

        loader.proque.set_dims(1 << 20);

        // Buffers
        let buffer_rhs = loader.proque.create_buffer::<T>().unwrap();
        let buffer_lhs = loader.proque.create_buffer::<T>().unwrap();
        let buffer_output = loader.proque.create_buffer::<T>().unwrap();

        buffer_rhs.write(&rhs.data).enq().unwrap();
        buffer_lhs.write(&self.data).enq().unwrap();

        // Build the kernel
        let kernel = match loader
            .proque
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
