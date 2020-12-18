use std::ffi::c_void;

use ykcompile::CompiledTrace;
use yktrace::sir::{SirTrace, SIR};
use yktrace::tir::TirTrace;

/// The different ways by which we can collect a trace.
#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
enum TracingKind {
    /// Software tracing via ykrustc.
    SoftwareTracing = 0,
    /// Hardware tracing via ykrustc + hwtracer.
    HardwareTracing = 1,
}

#[no_mangle]
unsafe extern "C" fn __ykrt_start_tracing(tracing_kind: u8) -> *mut c_void {
    Box::into_raw(Box::new(yktrace::start_tracing(match tracing_kind {
        0 => yktrace::TracingKind::SoftwareTracing,
        1 => yktrace::TracingKind::HardwareTracing,
        _ => std::process::abort(), // FIXME return error
    }))) as *mut c_void
}

#[no_mangle]
unsafe extern "C" fn __ykrt_stop_tracing(tracer: *mut c_void) -> *mut c_void {
    Box::into_raw(Box::new(
        Box::from_raw(tracer as *mut yktrace::ThreadTracer)
            .stop_tracing()
            .unwrap_or_else(|_| {
                std::process::abort(); // FIXME return error
            }),
    )) as *mut c_void
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compile_trace(sir_trace: *mut c_void) -> *mut c_void {
    let tt = TirTrace::new(&*SIR, &*Box::from_raw(sir_trace as *mut SirTrace)).unwrap();
    Box::into_raw(Box::new(ykcompile::compile_trace(tt))) as *mut c_void
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compiled_trace_get_ptr(compiled_trace: *const c_void) -> *const c_void {
    let compiled_trace = &*(compiled_trace as *mut CompiledTrace);
    compiled_trace.ptr() as *const c_void
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compiled_trace_dealloc(compiled_trace: *mut c_void) {
    Box::from_raw(compiled_trace as *mut CompiledTrace);
}
