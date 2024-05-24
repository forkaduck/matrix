use ocl::{
    builders::{ProQueBuilder, ProgramBuilder},
    ProQue,
};
use std::any::TypeId;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug)]
pub enum KernelLoaderEr {
    UnsupportedType,
    SrcDirError(io::Error),
    SrcReadError(io::Error),
    SrcDirEmpty,
}

pub struct KernelLoader {
    pub proque: ProQue,
}

impl KernelLoader {
    pub fn new<T: 'static>(kernel_dir: &Path) -> Result<Self, KernelLoaderEr> {
        let mut proque = ProQueBuilder::new();
        let mut src: Vec<String> = Vec::new();

        // Read all file contents into a vec.
        let directory_entries = match fs::read_dir(kernel_dir) {
            Ok(a) => a,
            Err(e) => {
                return Err(KernelLoaderEr::SrcDirError(e));
            }
        };

        for entry in directory_entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                let e_type = entry.file_type().expect("get file type");
                let e_extension = path
                    .extension()
                    .and_then(OsStr::to_str)
                    .expect("get file extension");

                if e_type.is_file() && e_extension == "cl" {
                    let file = match fs::read_to_string(path) {
                        Ok(a) => a,
                        Err(e) => {
                            return Err(KernelLoaderEr::SrcReadError(e));
                        }
                    };

                    src.push(file);
                }
            }
        }

        if src.is_empty() {
            return Err(KernelLoaderEr::SrcDirEmpty);
        }

        // Dynamically adjust types of kernels.
        let src_prefix = {
            use std::collections::HashMap;

            let map: HashMap<TypeId, &str> = [
                (TypeId::of::<f32>(), "float"),
                (TypeId::of::<f64>(), "double"),
            ]
            .into();

            match map.get(&TypeId::of::<T>()) {
                Some(a) => {
                    format!("#define FLOAT_T {}", a)
                }
                None => {
                    return Err(KernelLoaderEr::UnsupportedType);
                }
            }
        };

        // Add the source to the program and compile.
        let mut prog_build = ProgramBuilder::new();
        for i in &mut src {
            i.insert_str(0, &src_prefix);
            prog_build.source(i.clone());
        }

        let proque = match proque.prog_bldr(prog_build).build() {
            Ok(a) => a,
            Err(e) => {
                panic!("{}", e);
            }
        };

        Ok(KernelLoader { proque })
    }
}
