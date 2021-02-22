//! Client to the ykshim crate in the internal workspace.
//!
//! For more information, see this section in the documentation:
//! https://softdevteam.github.io/ykdocs/tech/yk_structure.html
//!
//! Put anything used only in testing in the `test_api` module.
//!
//! The exception to this rule is `Drop` implementations for opaque pointer wrappers. These should
//! always go in the `prod_api` module. It's hard to know all of the call sites for `drop()` since
//! they are implicit, so let's assume the production API does use them.

// FIXME handle all errors that may pass over the API boundary.

use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::{mem, ptr};

mod test_api;
pub use test_api::*;

// Opaque pointers.
pub(crate) type RawCompiledTrace = c_void;
pub(crate) type RawSirTrace = c_void;
type RawThreadTracer = c_void;
pub(crate) type RawTirTrace = c_void;
pub type RawStopgapInterpreter = c_void;

// These types and constants must be kept in sync with types of the same name in the internal
// workspace.
pub type LocalIndex = u32;
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
#[repr(C)]
pub struct Local(pub LocalIndex);
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TyIndex(pub u32);

extern "C" {
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
    fn __ykshim_si_interpret(interp: *mut RawStopgapInterpreter, ctx: *mut u8);
    fn __ykshim_sirinterpreter_drop(interp: *mut RawStopgapInterpreter);
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

pub struct StopgapInterpreter(pub *mut RawStopgapInterpreter);

impl StopgapInterpreter {
    pub unsafe fn interpret(&mut self, ctx: *mut u8) {
        __ykshim_si_interpret(self.0, ctx);
    }
}

impl Drop for StopgapInterpreter {
    fn drop(&mut self) {
        unsafe { __ykshim_sirinterpreter_drop(self.0) }
    }
}

pub struct SirTrace(pub(crate) *mut RawSirTrace);

unsafe impl Send for SirTrace {}
unsafe impl Sync for SirTrace {}

impl Drop for SirTrace {
    fn drop(&mut self) {
        if self.0 != ptr::null_mut() {
            unsafe { __ykshim_sirtrace_drop(self.0) }
        }
    }
}

pub struct CompiledTrace<I> {
    pub(crate) compiled: *mut c_void,
    pub(crate) _marker: PhantomData<I>,
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
    pub unsafe fn execute(&self, ctx: &mut I) -> *mut RawStopgapInterpreter {
        let f = mem::transmute::<_, fn(&mut I) -> *mut RawStopgapInterpreter>(self.ptr());
        f(ctx)
    }
}

impl<I> Drop for CompiledTrace<I> {
    fn drop(&mut self) {
        unsafe { __ykshim_compiled_trace_drop(self.compiled) }
    }
}
