use libc::size_t;
use std::convert::TryFrom;
use std::default::Default;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::ptr;

use ykbh::SIRInterpreter;
use ykcompile::{TraceCompiler, REG_POOL};
use ykpack::{self, CguHash, Local, LocalDecl, TyIndex};
use yktrace::sir::{self, SirTrace, SIR};
use yktrace::tir::TirTrace;

/// Returns the length of a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshim_sirtrace_len(sir_trace: *mut SirTrace) -> size_t {
    (&mut *sir_trace).len()
}

/// Compile a TIR trace from a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshim_tirtrace_new<'a, 'm>(
    sir_trace: *mut SirTrace,
) -> *mut TirTrace<'a, 'm> {
    let sir_trace = &mut *(sir_trace);
    Box::into_raw(Box::new(TirTrace::new(&SIR, sir_trace).unwrap()))
}

/// Returns the length of a SIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshim_tirtrace_len<'a, 'm>(tir_trace: *mut TirTrace<'a, 'm>) -> size_t {
    Box::from_raw(tir_trace).len()
}

/// Returns the human-readable Display string of a TIR trace.
#[no_mangle]
unsafe extern "C" fn __ykshim_tirtrace_display<'a, 'm>(
    tir_trace: *mut TirTrace<'a, 'm>,
) -> *mut c_char {
    let tt = Box::from_raw(tir_trace);
    let st = CString::new(format!("{}", tt)).unwrap();
    CString::into_raw(st)
}

/// Looks up the TypeId of the return value of the given symbol. The TypeId is returned in two
/// parts in `ret_cgu` and `ret_idx`.
#[no_mangle]
unsafe extern "C" fn __ykshim_body_ret_ty(
    sym: *mut c_char,
    ret_cgu: *mut CguHash,
    ret_idx: *mut TyIndex,
) {
    let sym = CString::from_raw(sym);
    let rv = usize::try_from(sir::RETURN_LOCAL.0).unwrap();
    let tyid = SIR.body(&sym.to_str().unwrap()).unwrap().local_decls[rv].ty;
    *ret_cgu = tyid.0;
    *ret_idx = tyid.1;
}

/// Creates a TraceCompiler with default settings.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_default() -> *mut TraceCompiler {
    let tc = Box::new(TraceCompiler::new(Default::default(), Default::default()));
    Box::into_raw(tc)
}

/// Drop a TraceCompiler.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_drop(comp: *mut c_void) {
    Box::from_raw(comp as *mut TraceCompiler);
}

/// Inserts a local declaration of the specified TypeId into a TraceCompiler. The TypeId is passed
/// in two parts: a CGU hash and a type index.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_insert_decl(
    tc: *mut c_void,
    local: Local,
    local_ty_cgu: CguHash,
    local_ty_index: TyIndex,
    referenced: bool,
) {
    let tc = &mut *(tc as *mut TraceCompiler);
    tc.local_decls.insert(
        local,
        LocalDecl {
            ty: (local_ty_cgu, local_ty_index),
            referenced,
        },
    );
}

/// Returns a string describing the register allocation of the specified local.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_local_to_location_str(
    tc: *mut c_void,
    local: Local,
) -> *mut c_char {
    let tc = &mut *(tc as *mut TraceCompiler);
    let rstr = format!("{:?}", tc.local_to_location(local));
    CString::new(rstr.as_str()).unwrap().into_raw()
}

/// Inform a TraceCompiler's register allocator that a local variable is dead.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_local_dead(tc: *mut TraceCompiler, local: Local) {
    let tc = &mut *tc;
    tc.local_dead(&local).unwrap();
}

/// Find a symbol's address in the current memory image. Returns NULL if it can't be found.
#[no_mangle]
unsafe extern "C" fn __ykshim_tracecompiler_find_sym(sym: *mut c_char) -> *mut c_void {
    TraceCompiler::find_symbol(CString::from_raw(sym).to_str().unwrap())
        .unwrap_or_else(|_| ptr::null_mut())
}

/// Interpret a SIR body with the specified interpreter context.
#[no_mangle]
unsafe extern "C" fn __yktest_interpret_body(body_name: *mut c_char, icx: *mut u8) {
    let body = SIR
        .body(CString::from_raw(body_name).to_str().unwrap())
        .unwrap();
    let mut si = SIRInterpreter::new(&*body);
    si.set_trace_inputs(icx);
    si.interpret(body);
}

/// Returns the size of the register allocators register pool.
#[no_mangle]
unsafe extern "C" fn __yktest_reg_pool_size() -> usize {
    REG_POOL.len()
}
