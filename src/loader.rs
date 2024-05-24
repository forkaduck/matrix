use ocl::{
    builders::{ProQueBuilder, ProgramBuilder},
    ProQue,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TypeMap {
    F32,
    F64,
}

impl TypeMap {
    fn c_str(&self) -> Option<&str> {
        let map: HashMap<TypeMap, &str> =
            [(TypeMap::F32, "float"), (TypeMap::F64, "double")].into();

        map.get(self).copied()
    }
}

impl TryFrom<&TypeId> for TypeMap {
    type Error = ();

    fn try_from(input: &TypeId) -> Result<Self, Self::Error> {
        let map: HashMap<TypeId, TypeMap> = [
            (TypeId::of::<f32>(), TypeMap::F32),
            (TypeId::of::<f64>(), TypeMap::F64),
        ]
        .into();

        match map.get(input) {
            Some(a) => Ok(a.clone()),
            None => Err(()),
        }
    }
}

#[derive(Debug)]
pub enum KernelLoaderEr {
    UnsupportedType,
    SrcDirError(io::Error),
    SrcReadError(io::Error),
    SrcDirEmpty,
}

#[allow(dead_code)]
pub struct KernelLoader {
    pub proque: ProQue,
    matrix_type: TypeMap,
}

impl KernelLoader {
    /// Loads and compiles all kernels.
    ///
    /// On success, returns a new KernelLoader. The object can then be used to create
    /// matrices using the matrix_new macro.
    ///
    /// * `kernel_dir` - The directory of all OpenCL C files (.cl).
    pub fn new<T: 'static>(kernel_dir: &Path) -> Result<Self, KernelLoaderEr> {
        let mut proque = ProQueBuilder::new();
        let mut src: Vec<String> = Vec::new();

        // Convert generic type to internal map of supported types.
        let matrix_type = match TypeMap::try_from(&TypeId::of::<T>()) {
            Ok(a) => a,
            Err(_) => {
                return Err(KernelLoaderEr::UnsupportedType);
            }
        };

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
            match matrix_type.c_str() {
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

        Ok(KernelLoader {
            proque,
            matrix_type,
        })
    }
}
