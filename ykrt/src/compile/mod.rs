use crate::{mt::MT, trace::IRTrace};
use libc::c_void;
use std::{collections::HashMap, error::Error, fmt, slice, sync::Arc};
use tempfile::NamedTempFile;
use yksmp::{LiveVar, StackMapParser};

#[cfg(jitc_llvm)]
pub(crate) mod jitc_llvm;

/// The trait that every JIT compiler backend must implement.
pub trait Compiler: Send + Sync {
    /// Compile an [IRTrace] into machine code.
    fn compile(
        &self,
        irtrace: IRTrace,
    ) -> Result<(*const c_void, Option<NamedTempFile>), Box<dyn Error>>;

    #[cfg(feature = "yk_testing")]
    unsafe fn compile_for_tc_tests(
        &self,
        irtrace: IRTrace,
        llvmbc_data: *const u8,
        llvmbc_len: u64,
    );
}

pub fn default_compiler() -> Result<Arc<dyn Compiler>, Box<dyn Error>> {
    #[cfg(jitc_llvm)]
    {
        return Ok(jitc_llvm::JITCLLVM::new()?);
    }

    #[allow(unreachable_code)]
    Err("No JIT compiler supported on this platform/configuration.".into())
}

struct SendSyncConstPtr<T>(*const T);
unsafe impl<T> Send for SendSyncConstPtr<T> {}
unsafe impl<T> Sync for SendSyncConstPtr<T> {}

struct Guard {
    failed: u32,
    code: Option<SendSyncConstPtr<c_void>>,
}

/// A trace compiled into machine code. Note that these are passed around as raw pointers and
/// potentially referenced by multiple threads so, once created, instances of this struct can only
/// be updated if a lock is held or a field is atomic.
pub struct CompiledTrace {
    pub mt: Arc<MT>,
    /// A function which when called, executes the compiled trace.
    ///
    /// The argument to the function is a pointer to a struct containing the live variables at the
    /// control point. The exact definition of this struct is not known to Rust: the struct is
    /// generated at interpreter compile-time by ykllvm.
    entry: SendSyncConstPtr<c_void>,
    /// Parsed stackmap of this trace. We only need to read this once, and can then use it to
    /// lookup stackmap information for each guard failure as needed.
    pub smap: HashMap<u64, Vec<LiveVar>>,
    /// Pointer to heap allocated live AOT values.
    aotvals: SendSyncConstPtr<c_void>,
    /// List of guards containing hotness counts or compiled side traces.
    guards: Vec<Option<Guard>>,
    /// If requested, a temporary file containing the "source code" for the trace, to be shown in
    /// debuggers when stepping over the JITted code.
    ///
    /// (rustc incorrectly identifies this field as dead code. Although it isn't being "used", the
    /// act of storing it is preventing the deletion of the file via its `Drop`)
    #[allow(dead_code)]
    di_tmpfile: Option<NamedTempFile>,
}

impl CompiledTrace {
    /// Create a `CompiledTrace` from a pointer to an array containing: the pointer to the compiled
    /// trace, the pointer to the stackmap and the size of the stackmap, and the pointer to the
    /// live AOT values.
    pub fn new(mt: Arc<MT>, data: *const c_void, di_tmpfile: Option<NamedTempFile>) -> Self {
        let slice = unsafe { slice::from_raw_parts(data as *const usize, 5) };
        let funcptr = slice[0] as *const c_void;
        let smptr = slice[1] as *const c_void;
        let smsize = slice[2];
        let aotvals = slice[3] as *mut c_void;
        let guardcount = slice[4] as usize;

        // Parse the stackmap of this trace and cache it. The original data allocated by memman.cc
        // is now no longer needed and can be freed.
        let smslice = unsafe { slice::from_raw_parts(smptr as *mut u8, smsize) };
        let smap = StackMapParser::parse(smslice).unwrap();
        unsafe { libc::munmap(smptr as *mut c_void, smsize) };

        // We heap allocated this array in yktracec to pass the data here. Now that we've
        // extracted it we no longer need to keep the array around.
        unsafe { libc::free(data as *mut c_void) };
        Self {
            mt,
            entry: SendSyncConstPtr(funcptr),
            smap,
            aotvals: SendSyncConstPtr(aotvals),
            di_tmpfile,
            guards: Vec::with_capacity(guardcount),
        }
    }

    #[cfg(any(test, feature = "yk_testing"))]
    #[doc(hidden)]
    /// Create a `CompiledTrace` with null contents. This is unsafe and only intended for testing
    /// purposes where a `CompiledTrace` instance is required, but cannot sensibly be constructed
    /// without overwhelming the test. The resulting instance must not be inspected or executed.
    pub unsafe fn new_null(mt: Arc<MT>) -> Self {
        Self {
            mt,
            entry: SendSyncConstPtr(std::ptr::null()),
            smap: HashMap::new(),
            aotvals: SendSyncConstPtr(std::ptr::null()),
            di_tmpfile: None,
            guards: Vec::new(),
        }
    }

    pub fn aotvals(&self) -> *const c_void {
        self.aotvals.0
    }

    pub fn entry(&self) -> *const c_void {
        self.entry.0
    }
}

impl Drop for CompiledTrace {
    fn drop(&mut self) {
        // The memory holding the AOT live values needs to live as long as the trace. Now that we
        // no longer need the trace, this can be freed too.
        unsafe { libc::free(self.aotvals.0 as *mut c_void) };
    }
}

impl fmt::Debug for CompiledTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompiledTrace {{ ... }}")
    }
}
