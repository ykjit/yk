use std::ffi::{c_void, CString};
use std::panic::UnwindSafe;
use std::os::raw::c_char;

use ykcompile::CompiledTrace;
use yktrace::sir::{SirTrace, SIR};
use yktrace::tir::TirTrace;

fn handle_panic<T>(f: impl FnOnce() -> T + UnwindSafe) -> T {
    std::panic::catch_unwind(|| f()).unwrap_or_else(|_| {
        // It is not safe for panics to cross through the "C" abi.
        // FIXME use "C unwind" as abi once it is implemented.
        std::process::abort();
    })
}

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
    handle_panic(|| {
        let tracing_kind = match tracing_kind {
            0 => yktrace::TracingKind::SoftwareTracing,
            1 => yktrace::TracingKind::HardwareTracing,
            _ => return std::ptr::null_mut(),
        };
        let tracer = yktrace::start_tracing(tracing_kind);
        Box::into_raw(Box::new(tracer)) as *mut c_void
    })
}

#[no_mangle]
unsafe extern "C" fn __ykrt_stop_tracing(
    tracer: *mut c_void,
    error_msg: *mut *mut c_char,
) -> *mut c_void {
    handle_panic(|| {
        let tracer = Box::from_raw(tracer as *mut yktrace::ThreadTracer);
        let sir_trace = tracer.stop_tracing();
        match sir_trace {
            Ok(sir_trace) => Box::into_raw(Box::new(sir_trace)) as *mut c_void,
            Err(err) => {
                *error_msg = CString::new(err.to_string())
                    .unwrap_or_else(|err| {
                        eprintln!("Stop tracing error {} contains a null byte", err);
                        std::process::abort();
                    })
                    .into_raw();
                std::ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compile_trace(
    sir_trace: *mut c_void,
    error_msg: *mut *mut c_char,
) -> *mut c_void {
    handle_panic(|| {
        let sir_trace = Box::from_raw(sir_trace as *mut SirTrace);

        let tt = match TirTrace::new(&*SIR, &*sir_trace) {
            Ok(tt) => tt,
            Err(err) => {
                *error_msg = CString::new(err.to_string())
                    .unwrap_or_else(|err| {
                        eprintln!("Tir compilation error {} contains a null byte", err);
                        std::process::abort();
                    })
                    .into_raw();
                return std::ptr::null_mut();
            }
        };
        let compiled_trace = ykcompile::compile_trace(tt);
        Box::into_raw(Box::new(compiled_trace)) as *mut c_void
    })
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compiled_trace_get_ptr(compiled_trace: *const c_void) -> *const c_void {
    let compiled_trace = &*(compiled_trace as *mut CompiledTrace);
    compiled_trace.ptr() as *const c_void
}

#[no_mangle]
unsafe extern "C" fn __ykrt_compiled_trace_dealloc(compiled_trace: *mut c_void) {
    handle_panic(|| Box::from_raw(compiled_trace as *mut CompiledTrace));
}
