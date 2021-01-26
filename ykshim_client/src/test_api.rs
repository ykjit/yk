//! The testing API client to ykshim.

use crate::prod_api::{Local, RawSirTrace, SirTrace, TyIndex};
use libc::size_t;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::fmt;
use std::os::raw::c_char;

// Opaque pointers.
type RawTirTrace = c_void;
type RawTraceCompiler = c_void;

// Keep these types in-sync with the internal workspace.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CguHash(u64);
// FIMXE make this a repr(C) struct so as to simplify __ykshimtest_body_ret_ty.
pub type TypeId = (CguHash, TyIndex);

extern "C" {
    fn __ykshimtest_sirtrace_len(sir_trace: *mut RawSirTrace) -> size_t;
    fn __ykshimtest_tirtrace_new(sir_trace: *mut RawSirTrace) -> *mut RawTirTrace;
    fn __ykshimtest_tracecompiler_drop(comp: *mut RawTraceCompiler);
    fn __ykshimtest_tirtrace_len(tir_trace: *mut RawTirTrace) -> size_t;
    fn __ykshimtest_tirtrace_display(tir_trace: *mut RawTirTrace) -> *mut c_char;
    fn __ykshimtest_body_ret_ty(sym: *mut c_char, ret_cgu: *mut CguHash, ret_idx: *mut TyIndex);
    fn __ykshimtest_tracecompiler_default() -> *mut RawTraceCompiler;
    fn __ykshimtest_tracecompiler_insert_decl(
        tc: *mut RawTraceCompiler,
        local: Local,
        local_ty_cgu: CguHash,
        local_ty_index: TyIndex,
        referenced: bool,
    );
    fn __ykshimtest_tracecompiler_local_to_location_str(
        tc: *mut RawTraceCompiler,
        local: Local,
    ) -> *mut c_char;
    fn __ykshimtest_tracecompiler_local_dead(tc: *mut RawTraceCompiler, local: Local);
    fn __ykshimtest_tracecompiler_find_sym(sym: *mut c_char) -> *mut c_void;
    fn __ykshimtest_interpret_body(body_name: *mut c_char, icx: *mut u8);
    fn __ykshimtest_reg_pool_size() -> usize;
}

pub struct TirTrace(*mut RawTirTrace);

impl TirTrace {
    pub fn new(sir_trace: &SirTrace) -> Self {
        Self(unsafe { __ykshimtest_tirtrace_new(sir_trace.0) })
    }

    pub fn len(&self) -> usize {
        unsafe { __ykshimtest_tirtrace_len(self.0) }
    }
}

impl fmt::Display for TirTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cstr = unsafe { CString::from_raw(__ykshimtest_tirtrace_display(self.0)) };
        write!(f, "{}", cstr.into_string().unwrap())
    }
}

pub fn sir_body_ret_ty(sym: &str) -> TypeId {
    let sym_c = CString::new(sym).unwrap();
    let mut cgu = CguHash(0);
    let mut idx = 0;
    unsafe { __ykshimtest_body_ret_ty(sym_c.into_raw(), &mut cgu, &mut idx) };
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
        let tc = unsafe { __ykshimtest_tracecompiler_default() };
        for (l, decl) in local_decls.iter() {
            unsafe {
                __ykshimtest_tracecompiler_insert_decl(
                    tc,
                    *l,
                    decl.ty.0,
                    decl.ty.1,
                    decl.referenced,
                )
            };
        }

        Self(tc)
    }

    pub fn local_to_location_str(&mut self, local: Local) -> String {
        let ptr = unsafe { __ykshimtest_tracecompiler_local_to_location_str(self.0, local) };
        String::from(unsafe { CString::from_raw(ptr).to_str().unwrap() })
    }

    pub fn local_dead(&mut self, local: Local) {
        unsafe { __ykshimtest_tracecompiler_local_dead(self.0, local) };
    }

    pub fn find_symbol(sym: &str) -> *mut c_void {
        let ptr = CString::new(sym).unwrap().into_raw();
        unsafe { __ykshimtest_tracecompiler_find_sym(ptr) }
    }
}

impl Drop for TraceCompiler {
    fn drop(&mut self) {
        unsafe { __ykshimtest_tracecompiler_drop(self.0) }
    }
}

impl SirTrace {
    pub fn len(&self) -> usize {
        unsafe { __ykshimtest_sirtrace_len(self.0) }
    }
}

pub fn interpret_body<I>(body_name: &str, icx: &mut I) {
    let body_cstr = CString::new(body_name).unwrap();
    unsafe { __ykshimtest_interpret_body(body_cstr.into_raw(), icx as *mut _ as *mut u8) };
}

pub fn reg_pool_size() -> usize {
    unsafe { __ykshimtest_reg_pool_size() }
}
