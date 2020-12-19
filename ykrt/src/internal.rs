use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::os::raw::c_char;

// Force ykrt_internal to be linked
#[allow(unused_imports)]
use ykrt_internal;

extern "C" {
    fn __ykrt_start_tracing(tracing_kind: u8) -> *mut c_void;
    fn __ykrt_stop_tracing(tracer: *mut c_void, error_msg: *mut *mut c_char) -> *mut c_void;
    fn __ykrt_compile_trace(sir_trace: *mut c_void, error_msg: *mut *mut c_char) -> *mut c_void;
    fn __ykrt_compiled_trace_get_ptr(compiled_trace: *const c_void) -> *const c_void;
    fn __ykrt_compiled_trace_dealloc(compiled_trace: *mut c_void);
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

pub(crate) struct ThreadTracer(*mut c_void);

pub(crate) fn start_tracing(tracing_kind: TracingKind) -> ThreadTracer {
    let tracer = unsafe { __ykrt_start_tracing(tracing_kind as u8) };
    assert!(!tracer.is_null());
    ThreadTracer(tracer)
}

impl ThreadTracer {
    pub(crate) fn stop_tracing(self) -> Result<SirTrace, CString> {
        let mut err_msg = std::ptr::null_mut();
        let sir_trace = unsafe { __ykrt_stop_tracing(self.0, &mut err_msg) };
        if sir_trace.is_null() {
            return Err(unsafe { CString::from_raw(err_msg) });
        }
        Ok(SirTrace(sir_trace))
    }
}

pub(crate) struct SirTrace(*mut c_void);

unsafe impl Send for SirTrace {}
unsafe impl Sync for SirTrace {}

pub(crate) struct CompiledTrace<I> {
    compiled: *mut c_void,
    _marker: PhantomData<I>,
}

unsafe impl<I> Send for CompiledTrace<I> {}
unsafe impl<I> Sync for CompiledTrace<I> {}

pub(crate) fn compile_trace<T>(sir_trace: SirTrace) -> Result<CompiledTrace<T>, CString> {
    let mut err_msg = std::ptr::null_mut();
    let compiled = unsafe { __ykrt_compile_trace(sir_trace.0, &mut err_msg) };
    if compiled.is_null() {
        return Err(unsafe { CString::from_raw(err_msg) });
    }
    Ok(CompiledTrace {
        compiled,
        _marker: PhantomData,
    })
}

impl<I> CompiledTrace<I> {
    pub(crate) fn ptr(&self) -> *const u8 {
        unsafe { __ykrt_compiled_trace_get_ptr(self.compiled) as *const u8 }
    }
}

impl<I> Drop for CompiledTrace<I> {
    fn drop(&mut self) {
        unsafe { __ykrt_compiled_trace_dealloc(self.compiled) }
    }
}
