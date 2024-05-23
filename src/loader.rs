use ocl::{
    builders::{ProQueBuilder, ProgramBuilder},
    ProQue,
};
use std::ffi::OsStr;
use std::fs;
use std::path::Path;

pub struct KernelLoader {
    pub proque: ProQue,
}

impl KernelLoader {
    pub fn new(kernel_dir: &Path) -> Self {
        let mut proque = ProQueBuilder::new();
        let mut src: Vec<String> = Vec::new();

        // Read all file contents into a vec.
        for entry in fs::read_dir(kernel_dir).expect("read source directory") {
            if let Ok(entry) = entry {
                let path = entry.path();
                let e_type = entry.file_type().expect("get file type");
                let e_extension = path
                    .extension()
                    .and_then(OsStr::to_str)
                    .expect("get file extension");

                if e_type.is_file() && e_extension == "cl" {
                    src.push(fs::read_to_string(path).expect("read source file"));
                }
            }
        }

        if src.is_empty() {
            panic!("No source files to compile!");
        }

        // Add the source to the program and compile.
        let mut prog_build = ProgramBuilder::new();
        for i in src {
            prog_build.source(i);
        }

        let proque = match proque.prog_bldr(prog_build).build() {
            Ok(a) => a,
            Err(e) => {
                println!("OpenCL Kernel Compile Error: {}", e);
                panic!();
            }
        };

        KernelLoader { proque }
    }
}
