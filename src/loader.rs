use half::f16;
use log::debug;
use ocl::{
    builders::{BuildOpt, ProgramBuilder},
    enums::{DeviceInfo, DeviceInfoResult},
    flags::{DeviceFpConfig, DeviceType},
    Buffer, Context, Device, Kernel, Platform, Program, Queue, SpatialDims,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

/// TypeMap is an internal type map which represents all possible types
/// useable by the compute shaders.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TypeMap {
    F16,
    F32,
    F64,
}

impl TypeMap {
    fn c_str(&self) -> &str {
        match self {
            TypeMap::F16 => "half",
            TypeMap::F32 => "float",
            TypeMap::F64 => "double",
        }
    }

    fn from_typeid(input: &TypeId) -> TypeMap {
        let map: HashMap<TypeId, TypeMap> = [
            (TypeId::of::<f16>(), TypeMap::F16),
            (TypeId::of::<f32>(), TypeMap::F32),
            (TypeId::of::<f64>(), TypeMap::F64),
        ]
        .into();

        map.get(input)
            .copied()
            .expect("Missing from_typeid implementation (bug)")
    }
}

pub struct KernelType {
    static_repr: TypeMap,
    fp_config: DeviceFpConfig,
}

impl KernelType {
    fn new(type_id: &TypeId, dev: &Device) -> Option<KernelType> {
        let static_repr = TypeMap::from_typeid(type_id);

        let fp_config = match static_repr {
            TypeMap::F16 => match dev.info(DeviceInfo::HalfFpConfig).expect("no HalfFpConfig") {
                DeviceInfoResult::HalfFpConfig(a) => a,
                _ => return None,
            },
            TypeMap::F32 => match dev
                .info(DeviceInfo::SingleFpConfig)
                .expect("no SingleFpConfig")
            {
                DeviceInfoResult::SingleFpConfig(a) => a,
                _ => return None,
            },
            TypeMap::F64 => match dev
                .info(DeviceInfo::DoubleFpConfig)
                .expect("no DoubleFpConfig")
            {
                DeviceInfoResult::DoubleFpConfig(a) => a,
                _ => return None,
            },
        };

        Some(KernelType {
            static_repr,
            fp_config,
        })
    }

    pub fn get_fp_config(&self) -> DeviceFpConfig {
        self.fp_config
    }

    pub fn get_type(&self) -> TypeMap {
        self.static_repr
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
    SrcDirReadError(io::Error),
    SrcDirError,
    SrcReadError(io::Error),
    SrcDirEmpty,
    PlatformError(ocl::error::Error),
    QueueError(ocl::error::Error),
    ContextError(ocl::error::Error),
}

#[allow(dead_code)]
pub struct KernelLoader {
    pub global_work_size: SpatialDims,
    pub local_work_size: SpatialDims,

    pub context: Context,
    pub queue: Queue,
    pub program: Program,

    pub kernel_type: KernelType,
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

    /// Checks for available devices
    ///
    /// Currently, goes through all devices and searches the one with
    /// the highest "performance metrics".
    fn get_device() -> Result<(Platform, Device), KernelLoaderEr> {
        let mut device_list: HashMap<u64, (Platform, Device)> = HashMap::new();

        for pl in Platform::list() {
            match Device::list(pl, Some(DeviceType::GPU)) {
                Ok(a) => {
                    for dev in a {
                        let mut pref: u64 = 1;

                        if let DeviceInfoResult::MaxComputeUnits(temp) = dev
                            .info(DeviceInfo::MaxComputeUnits)
                            .expect("no MaxComputeUnits")
                        {
                            pref *= temp as u64;
                        }

                        if let DeviceInfoResult::MaxClockFrequency(temp) = dev
                            .info(DeviceInfo::MaxClockFrequency)
                            .expect("no MaxClockFrequency")
                        {
                            pref *= temp as u64;
                        }

                        pref *= dev.max_wg_size().expect("no MaxWorkGroupSize") as u64;
                        pref *= if dev.is_available().expect("no IsAvailable") {
                            1
                        } else {
                            0
                        };

                        device_list.insert(pref, (pl, dev));
                    }
                }
                Err(e) => return Err(KernelLoaderEr::PlatformError(e)),
            };
        }

        let mut last_pref = 1;
        for (pref, _) in &device_list {
            if *pref > last_pref {
                last_pref = *pref;
            }
        }

        Ok(device_list.get(&last_pref).unwrap().to_owned())
    }

    /// Runs a test kernel.
    ///
    /// Will probably be used in the future to detect various device
    /// specifics, or benchmark different capabilities.
    ///
    /// * `self` - A kernel loader object to operate on.
    fn run_test_kernel(&mut self) {
        const SIZE: usize = 10;
        let buffer_output = Buffer::<u8>::builder()
            .len(SIZE)
            .queue(self.queue.clone())
            .build()
            .expect("buffer out");

        let kernel = match Kernel::builder()
            .program(&self.program)
            .name("test_capabilities")
            .queue(self.queue.clone())
            .global_work_size(self.global_work_size)
            .local_work_size(self.local_work_size)
            .arg(&buffer_output)
            .build()
        {
            Ok(a) => a,
            Err(e) => {
                panic!("Test Kernel failed to compile!\n{}", e);
            }
        };

        unsafe {
            kernel.cmd().queue(&self.queue).enq().expect("kernel enque");
        }

        let mut output: Vec<u8> = Vec::new();

        // Has to be preallocated
        output.resize(10, 0);

        buffer_output
            .read(&mut output)
            .enq()
            .expect("read from out");

        debug!("Test kernel output: {:?}", output);
    }

    /// Loads and compiles all kernels.
    ///
    /// On success, returns a new KernelLoader. The object can then be used to create
    /// matrices using the matrix_new macro.
    ///
    /// * `kernel_dir` - The directory of all OpenCL C files (.cl).
    pub fn new<T: 'static>(
        kernel_dir: &Path,
        unsafe_fast_math: bool,
        kernel_debug: bool,
        threads: usize,
    ) -> Result<Self, KernelLoaderEr> {
        let mut src: HashMap<String, String> = HashMap::new();

        let kernel_dir_str = match kernel_dir.to_str() {
            Some(a) => a,
            None => return Err(KernelLoaderEr::SrcDirError),
        };

        // Get the fastest device that is available.
        let (platfrom, device) = KernelLoader::get_device()?;
        debug!("Picked OpenCL device: {}", device.name().unwrap());

        let global_work_size =
            SpatialDims::from(device.max_wg_size().expect("no MaxWorkGroupSize") / threads);

        let local_work_size =
            SpatialDims::from(device.max_wg_size().expect("no MaxWorkGroupSize") / threads);

        // Construct a dynamic representation of the primary type used in all generic kernels.
        let kernel_type = match KernelType::new(&TypeId::of::<T>(), &device) {
            Some(a) => a,
            None => return Err(KernelLoaderEr::UnsupportedType),
        };

        debug!(
            "Device rounding mode with {:?}: {:?}",
            kernel_type.get_type(),
            kernel_type.get_fp_config()
        );

        // Read all file contents into a vec.
        let directory_entries = match fs::read_dir(kernel_dir) {
            Ok(a) => a,
            Err(e) => {
                return Err(KernelLoaderEr::SrcDirReadError(e));
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

        let mut prog_build = ProgramBuilder::new();

        prog_build.bo(BuildOpt::CmplrInclDir {
            path: kernel_dir_str.to_string(),
        });

        // Dynamically adjust types of kernels.
        let mut src_global_prefix = String::new();
        src_global_prefix.push_str(
            format!(
                "#undef TYPE_T\n#define TYPE_T {}\n",
                kernel_type.get_type().c_str()
            )
            .as_str(),
        );

        // Add compiler options for faster math.
        if unsafe_fast_math {
            prog_build.cmplr_opt("-cl-finite-math-only -cl-unsafe-math-optimizations");
        }

        if kernel_debug {
            src_global_prefix.push_str(format!("#define DEBUG\n").as_str());
        }

        // Dynamically adjust the operator used in the kernel.
        for (idx, cs) in &mut src {
            cs.insert_str(0, &src_global_prefix);

            let current_variant = Self::get_variants().get(idx.as_str());

            // Is the current kernel a generic one?
            if cs.contains("OPERATOR")
                && cs.contains("KERNEL_NAME")
                && let Some(var) = current_variant
            {
                debug!("Found operator-generic kernel in {}", idx);

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
                debug!("Found generic kernel in {}", idx);

                prog_build.source(cs.clone());
            }
        }

        // Initialize the context, queue and program (which compiles the program).
        let context = match Context::builder().platform(platfrom).build() {
            Ok(a) => a,
            Err(e) => return Err(KernelLoaderEr::ContextError(e)),
        };

        let queue = match Queue::new(&context, device, None) {
            Ok(a) => a,
            Err(e) => return Err(KernelLoaderEr::QueueError(e)),
        };

        let program = match prog_build.devices(device).build(&context) {
            Ok(a) => a,
            Err(e) => panic!("\nSource file failed to compile!:{}", e),
        };

        let mut loader = KernelLoader {
            global_work_size,
            local_work_size,

            context,
            queue,
            program,

            kernel_type,
        };

        // Run the test kernel.
        loader.run_test_kernel();

        Ok(loader)
    }
}
