//! Client to the ykshim crate in the internal workspace.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html

use libc::size_t;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::fmt;
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::{mem, ptr};

// FIXME handle all errors that may pass over the API boundary.

// These types and constants must be kept in sync with types of the same name in the internal
// workspace.
pub type LocalIndex = u32;
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
#[repr(C)]
pub struct Local(pub LocalIndex);
pub type TyIndex = u32;
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CguHash(u64);
// FIMXE make this a repr(C) struct so as to simplify __ykshim_body_ret_ty.
pub type TypeId = (CguHash, TyIndex);
pub const RETURN_LOCAL: Local = Local(0);

// Opaque pointers.
type RawThreadTracer = c_void;
type RawSirTrace = c_void;
type RawTirTrace = c_void;
type RawCompiledTrace = c_void;
type RawTraceCompiler = c_void;

extern "C" {
    // The "Production" API.
    fn __ykshim_start_tracing(tracing_kind: u8) -> *mut RawThreadTracer;
    fn __ykshim_stop_tracing(
        tracer: *mut RawThreadTracer,
        error_msg: *mut *mut c_char,
    ) -> *mut RawSirTrace;
    fn __ykshim_compile_trace(
        sir_trace: *mut RawSirTrace,
        error_msg: *mut *mut c_char,
    ) -> *mut RawCompiledTrace;
    fn __ykshim_compiled_trace_get_ptr(compiled_trace: *const RawCompiledTrace) -> *const c_void;
    fn __ykshim_compiled_trace_drop(compiled_trace: *mut RawCompiledTrace);
    fn __ykshim_sirtrace_drop(trace: *mut RawSirTrace);

    // The testing API.
    fn __ykshim_sirtrace_len(sir_trace: *mut RawSirTrace) -> size_t;
    fn __ykshim_tirtrace_new(sir_trace: *mut RawSirTrace) -> *mut RawTirTrace;
    fn __ykshim_tracecompiler_drop(comp: *mut RawTraceCompiler);
    fn __ykshim_tirtrace_len(tir_trace: *mut RawTirTrace) -> size_t;
    fn __ykshim_tirtrace_display(tir_trace: *mut RawTirTrace) -> *mut c_char;
    fn __ykshim_body_ret_ty(sym: *mut c_char, ret_cgu: *mut CguHash, ret_idx: *mut TyIndex);
    fn __ykshim_tracecompiler_default() -> *mut RawTraceCompiler;
    fn __ykshim_tracecompiler_insert_decl(
        tc: *mut RawTraceCompiler,
        local: Local,
        local_ty_cgu: CguHash,
        local_ty_index: TyIndex,
        referenced: bool,
    );
    fn __ykshim_tracecompiler_local_to_location_str(
        tc: *mut RawTraceCompiler,
        local: Local,
    ) -> *mut c_char;
    fn __ykshim_tracecompiler_local_dead(tc: *mut RawTraceCompiler, local: Local);
    fn __ykshim_tracecompiler_find_sym(sym: *mut c_char) -> *mut c_void;
    fn __yktest_interpret_body(body_name: *mut c_char, icx: *mut u8);
    fn __yktest_reg_pool_size() -> usize;
}

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing = 0,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing = 1,
}

pub struct ThreadTracer(*mut RawThreadTracer);

/// Start tracing using the specified kind of tracing.
pub fn start_tracing(tracing_kind: TracingKind) -> ThreadTracer {
    let tracer = unsafe { __ykshim_start_tracing(tracing_kind as u8) };
    assert!(!tracer.is_null());
    ThreadTracer(tracer)
}

impl ThreadTracer {
    pub fn stop_tracing(mut self) -> Result<SirTrace, CString> {
        let mut err_msg = std::ptr::null_mut();
        let sir_trace = unsafe { __ykshim_stop_tracing(self.0, &mut err_msg) };
        self.0 = ptr::null_mut(); // consumed.
        if sir_trace.is_null() {
            return Err(unsafe { CString::from_raw(err_msg) });
        }
        Ok(SirTrace(sir_trace))
    }
}

impl Drop for ThreadTracer {
    fn drop(&mut self) {
        if self.0 != ptr::null_mut() {
            // We are still tracing.
            let mut err_msg = std::ptr::null_mut();
            unsafe { __ykshim_stop_tracing(self.0, &mut err_msg) };
        }
    }
}

pub struct SirTrace(*mut RawSirTrace);

unsafe impl Send for SirTrace {}
unsafe impl Sync for SirTrace {}

impl SirTrace {
    pub fn len(&self) -> usize {
        unsafe { __ykshim_sirtrace_len(self.0) }
    }
}

impl Drop for SirTrace {
    fn drop(&mut self) {
        if self.0 != ptr::null_mut() {
            unsafe { __ykshim_sirtrace_drop(self.0) }
        }
    }
}

pub struct TirTrace(*mut RawTirTrace);

impl TirTrace {
    pub fn new(sir_trace: &SirTrace) -> Self {
        Self(unsafe { __ykshim_tirtrace_new(sir_trace.0) })
    }

    pub fn len(&self) -> usize {
        unsafe { __ykshim_tirtrace_len(self.0) }
    }
}

impl fmt::Display for TirTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cstr = unsafe { CString::from_raw(__ykshim_tirtrace_display(self.0)) };
        write!(f, "{}", cstr.into_string().unwrap())
    }
}

pub struct CompiledTrace<I> {
    compiled: *mut c_void,
    _marker: PhantomData<I>,
}

unsafe impl<I> Send for CompiledTrace<I> {}
unsafe impl<I> Sync for CompiledTrace<I> {}

pub fn compile_trace<T>(mut sir_trace: SirTrace) -> Result<CompiledTrace<T>, CString> {
    let mut err_msg = std::ptr::null_mut();
    let compiled = unsafe { __ykshim_compile_trace(sir_trace.0, &mut err_msg) };
    sir_trace.0 = ptr::null_mut(); // consumed.
    if compiled.is_null() {
        return Err(unsafe { CString::from_raw(err_msg) });
    }
    Ok(CompiledTrace {
        compiled,
        _marker: PhantomData,
    })
}

impl<I> CompiledTrace<I> {
    pub fn ptr(&self) -> *const u8 {
        unsafe { __ykshim_compiled_trace_get_ptr(self.compiled) as *const u8 }
    }

    /// Execute the trace with the given interpreter context.
    pub unsafe fn execute(&self, icx: &mut I) -> bool {
        let f = mem::transmute::<_, fn(&mut I) -> bool>(self.ptr());
        f(icx)
    }
}

impl<I> Drop for CompiledTrace<I> {
    fn drop(&mut self) {
        unsafe { __ykshim_compiled_trace_drop(self.compiled) }
    }
}

pub fn sir_body_ret_ty(sym: &str) -> TypeId {
    let sym_c = CString::new(sym).unwrap();
    let mut cgu = CguHash(0);
    let mut idx = 0;
    unsafe { __ykshim_body_ret_ty(sym_c.into_raw(), &mut cgu, &mut idx) };
    (cgu, idx)
}

pub struct LocalDecl {
    pub ty: TypeId,
    pub referenced: bool,
}

impl LocalDecl {
    pub fn new(ty: TypeId, referenced: bool) -> Self {
        Self { ty, referenced }
    }
}

pub struct TraceCompiler(*mut c_void);

impl TraceCompiler {
    pub fn new(local_decls: HashMap<Local, LocalDecl>) -> Self {
        let tc = unsafe { __ykshim_tracecompiler_default() };
        for (l, decl) in local_decls.iter() {
            unsafe {
                __ykshim_tracecompiler_insert_decl(tc, *l, decl.ty.0, decl.ty.1, decl.referenced)
            };
        }

        Self(tc)
    }

    pub fn local_to_location_str(&mut self, local: Local) -> String {
        let ptr = unsafe { __ykshim_tracecompiler_local_to_location_str(self.0, local) };
        String::from(unsafe { CString::from_raw(ptr).to_str().unwrap() })
    }

    pub fn local_dead(&mut self, local: Local) {
        unsafe { __ykshim_tracecompiler_local_dead(self.0, local) };
    }

    pub fn find_symbol(sym: &str) -> *mut c_void {
        let ptr = CString::new(sym).unwrap().into_raw();
        unsafe { __ykshim_tracecompiler_find_sym(ptr) }
    }
}

impl Drop for TraceCompiler {
    fn drop(&mut self) {
        unsafe { __ykshim_tracecompiler_drop(self.0) }
    }
}

pub fn interpret_body<I>(body_name: &str, icx: &mut I) {
    let body_cstr = CString::new(body_name).unwrap();
    unsafe { __yktest_interpret_body(body_cstr.into_raw(), icx as *mut _ as *mut u8) };
}

pub fn reg_pool_size() -> usize {
    unsafe { __yktest_reg_pool_size() }
}
