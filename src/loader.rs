use log::debug;
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
use std::sync::OnceLock;

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

struct KernelVariant<'a> {
    pub name: &'a [&'a str],
    pub operator: &'a [&'a str],
    pub length: usize,
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
    fn get_variants() -> &'static HashMap<&'static str, KernelVariant<'static>> {
        static MAP: OnceLock<HashMap<&str, KernelVariant<'static>>> = OnceLock::new();

        MAP.get_or_init(|| {
            let mut m = HashMap::new();

            m.insert(
                "vec_arithmetic.cl",
                KernelVariant {
                    operator: &["+", "-", "*", "/"],
                    name: &["add", "sub", "mul", "div"],
                    length: 4,
                },
            );

            m
        })
    }

    /// Loads and compiles all kernels.
    ///
    /// On success, returns a new KernelLoader. The object can then be used to create
    /// matrices using the matrix_new macro.
    ///
    /// * `kernel_dir` - The directory of all OpenCL C files (.cl).
    pub fn new<T: 'static>(kernel_dir: &Path) -> Result<Self, KernelLoaderEr> {
        let mut proque = ProQueBuilder::new();
        let mut src: HashMap<String, String> = HashMap::new();

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

                let e_name = path
                    .file_name()
                    .expect("file name")
                    .to_str()
                    .expect("file name encoding");

                if e_type.is_file() && e_extension == "cl" {
                    let file = match fs::read_to_string(&path) {
                        Ok(a) => a,
                        Err(e) => {
                            return Err(KernelLoaderEr::SrcReadError(e));
                        }
                    };

                    src.insert(e_name.to_owned(), file);
                }
            }
        }

        if src.is_empty() {
            return Err(KernelLoaderEr::SrcDirEmpty);
        }

        debug!("Found {} source files", src.len());

        let mut src_global_prefix = String::new();

        // Dynamically adjust types of kernels.
        match matrix_type.c_str() {
            Some(a) => src_global_prefix.push_str(format!("#define FLOAT_T {}\n", a).as_str()),
            None => {
                return Err(KernelLoaderEr::UnsupportedType);
            }
        };

        let mut prog_build = ProgramBuilder::new();
        for (idx, cs) in &mut src {
            cs.insert_str(0, &src_global_prefix);

            let current_variant = Self::get_variants().get(idx.as_str());

            // Is the current kernel a generic one?
            if cs.contains("OPERATOR")
                && cs.contains("KERNEL_NAME")
                && let Some(var) = current_variant
            {
                debug!("Found generic kernel in {}", idx);

                for k in 0..var.length {
                    let mut cs_local = cs.clone();

                    cs_local
                        .insert_str(0, format!("#define KERNEL_NAME {}\n", var.name[k]).as_str());
                    cs_local.insert_str(
                        0,
                        format!("#define OPERATOR {}\n", var.operator[k]).as_str(),
                    );

                    // Is backwards because we insert at the top.
                    cs_local.insert_str(0, "#undef KERNEL_NAME\n#undef OPERATOR\n");

                    prog_build.source(cs_local.clone());
                }
            } else {
                debug!("Found kernel in {}", idx);

                prog_build.source(cs.clone());
            }
        }

        // Add the source to the program and compile.
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
