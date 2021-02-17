use libc::size_t;
use std::convert::TryFrom;
use std::default::Default;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use ykbh::SIRInterpreter;
use ykcompile::{find_symbol, CompiledTrace, TraceCompiler, REG_POOL};
use ykpack::{self, Local, LocalDecl, TypeId};
use yktrace::sir::{self, SirTrace, SIR};
use yktrace::tir::TirTrace;

/// Returns the length of a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_sirtrace_len(sir_trace: *mut SirTrace) -> size_t {
    (&mut *sir_trace).len()
}

/// Compile a TIR trace from a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tirtrace_new<'a, 'm>(
    sir_trace: *mut SirTrace,
) -> *mut TirTrace<'a, 'm> {
    let sir_trace = &mut *(sir_trace);
    Box::into_raw(Box::new(TirTrace::new(&SIR, sir_trace).unwrap()))
}

/// Returns the length of a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tirtrace_len<'a, 'm>(tir_trace: *mut TirTrace<'a, 'm>) -> size_t {
    (*tir_trace).len()
}

/// Returns the human-readable Display string of a TIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tirtrace_display<'a, 'm>(
    tir_trace: *mut TirTrace<'a, 'm>,
) -> *mut c_char {
    let tt = &(*tir_trace);
    let st = CString::new(format!("{}", tt)).unwrap();
    CString::into_raw(st)
}

/// Looks up the TypeId of the return value of the given symbol. The TypeId is returned via the
/// `ret_tyid` argument.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_body_ret_ty(sym: *const c_char, ret_tyid: *mut TypeId) {
    let sym = CStr::from_ptr(sym);
    let rv = usize::try_from(sir::RETURN_LOCAL.0).unwrap();
    let tyid = SIR.body(&sym.to_str().unwrap()).unwrap().local_decls[rv].ty;
    *ret_tyid = tyid;
}

/// Creates a TraceCompiler with default settings.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tracecompiler_default() -> *mut TraceCompiler {
    let tc = Box::new(TraceCompiler::new(Default::default(), Default::default()));
    Box::into_raw(tc)
}

/// Drop a TraceCompiler.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tracecompiler_drop(comp: *mut c_void) {
    Box::from_raw(comp as *mut TraceCompiler);
}

/// Inserts a local declaration of the specified TypeId into a TraceCompiler.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tracecompiler_insert_decl(
    tc: *mut c_void,
    local: Local,
    local_ty: TypeId,
    referenced: bool,
) {
    let tc = &mut *(tc as *mut TraceCompiler);
    tc.local_decls.insert(
        local,
        LocalDecl {
            ty: local_ty,
            referenced,
        },
    );
}

/// Returns a string describing the register allocation of the specified local.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tracecompiler_local_to_location_str(
    tc: *mut c_void,
    local: Local,
) -> *mut c_char {
    let tc = &mut *(tc as *mut TraceCompiler);
    let rstr = format!("{:?}", tc.local_to_location(local));
    CString::new(rstr.as_str()).unwrap().into_raw()
}

/// Inform a TraceCompiler's register allocator that a local variable is dead.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_tracecompiler_local_dead(tc: *mut TraceCompiler, local: Local) {
    let tc = &mut *tc;
    tc.local_dead(&local).unwrap();
}

/// Find a symbol's address in the current memory image. Returns NULL if it can't be found.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_find_symbol(sym: *const c_char) -> *mut c_void {
    find_symbol(CStr::from_ptr(sym).to_str().unwrap()).unwrap_or_else(|_| ptr::null_mut())
}

/// Interpret a SIR body with the specified interpreter context.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_interpret_body(body_name: *const c_char, ctx: *mut u8) {
    let fname = CStr::from_ptr(body_name).to_str().unwrap().to_string();
    let mut si = SIRInterpreter::from_symbol(fname);
    si.set_interp_ctx(ctx);
    si.interpret();
}

/// Returns the size of the register allocators register pool.
#[no_mangle]
unsafe extern "C" fn __ykshimtest_reg_pool_size() -> usize {
    REG_POOL.len()
}

/// Consumes and compiles the given TIR trace to native code, returning an opaque pointer to the
/// compiled trace.
#[no_mangle]
fn __ykshimtest_compile_tir_trace(tir_trace: *mut TirTrace) -> *mut CompiledTrace {
    let tir_trace = unsafe { Box::from_raw(tir_trace) };
    let compiled_trace = ykcompile::compile_trace(*tir_trace);
    Box::into_raw(Box::new(compiled_trace))
}
