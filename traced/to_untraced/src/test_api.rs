//! The testing interface to untraced_api.
#![cfg(feature = "testing")]

use crate::{CompiledTrace, Local, RawCompiledTrace, RawSirTrace, RawTirTrace, SirTrace, TyIndex};
use libc::size_t;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::{fmt, ptr};

type RawTraceCompiler = c_void;

// Keep these types in-sync with the untraced workspace.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CguHash(u64);
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[cfg(feature = "testing")]
pub struct TypeId {
    cgu: CguHash,
    idx: TyIndex,
}

extern "C" {
    fn __untraced_apitest_compile_tir_trace(tir_trace: *mut RawTirTrace) -> *mut RawCompiledTrace;
    fn __untraced_apitest_sirtrace_len(sir_trace: *mut RawSirTrace) -> size_t;
    fn __untraced_apitest_tirtrace_new(sir_trace: *mut RawSirTrace) -> *mut RawTirTrace;
    fn __untraced_api_tirtrace_drop(tir_trace: *mut RawTirTrace);
    fn __untraced_apitest_tracecompiler_drop(comp: *mut RawTraceCompiler);
    fn __untraced_apitest_tirtrace_len(tir_trace: *mut RawTirTrace) -> size_t;
    fn __untraced_apitest_tirtrace_display(tir_trace: *mut RawTirTrace) -> *mut c_char;
    fn __untraced_apitest_body_ret_ty(sym: *const c_char, ret_tyid: *mut TypeId);
    fn __untraced_apitest_tracecompiler_default() -> *mut RawTraceCompiler;
    fn __untraced_apitest_tracecompiler_insert_decl(
        tc: *mut RawTraceCompiler,
        local: Local,
        local_ty: TypeId,
        referenced: bool,
    );
    fn __untraced_apitest_tracecompiler_local_to_location_str(
        tc: *mut RawTraceCompiler,
        local: Local,
    ) -> *mut c_char;
    fn __untraced_apitest_tracecompiler_local_dead(tc: *mut RawTraceCompiler, local: Local);
    fn __untraced_apitest_find_symbol(sym: *const c_char) -> *mut c_void;
    fn __untraced_apitest_interpret_body(body_name: *const c_char, ctx: *mut u8);
    fn __untraced_apitest_reg_pool_size() -> usize;
}

#[derive(Debug)]
#[cfg(feature = "testing")]
pub struct TirTrace(*mut RawTirTrace);

impl TirTrace {
    pub fn new(sir_trace: &SirTrace) -> Self {
        Self(unsafe { __untraced_apitest_tirtrace_new(sir_trace.0) })
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        unsafe { __untraced_apitest_tirtrace_len(self.0) }
    }
}

impl Drop for TirTrace {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { __untraced_api_tirtrace_drop(self.0) };
        }
    }
}

impl fmt::Display for TirTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cstr = unsafe { CString::from_raw(__untraced_apitest_tirtrace_display(self.0)) };
        write!(f, "{}", cstr.into_string().unwrap())
    }
}

pub fn sir_body_ret_ty(sym: &str) -> TypeId {
    let sym_c = CString::new(sym).unwrap();
    let mut ret = TypeId {
        cgu: CguHash(0),
        idx: TyIndex(0),
    };
    unsafe { __untraced_apitest_body_ret_ty(sym_c.as_ptr(), &mut ret) };
    ret
}

pub struct LocalDecl {
    ty: TypeId,
    referenced: bool,
}

impl LocalDecl {
    pub fn new(ty: TypeId, referenced: bool) -> Self {
        Self { ty, referenced }
    }
}

pub struct TraceCompiler(*mut c_void);

impl TraceCompiler {
    pub fn new(local_decls: HashMap<Local, LocalDecl>) -> Self {
        let tc = unsafe { __untraced_apitest_tracecompiler_default() };
        for (l, decl) in local_decls.iter() {
            unsafe {
                __untraced_apitest_tracecompiler_insert_decl(tc, *l, decl.ty, decl.referenced)
            };
        }

        Self(tc)
    }

    pub fn local_to_location_str(&mut self, local: Local) -> String {
        let ptr = unsafe { __untraced_apitest_tracecompiler_local_to_location_str(self.0, local) };
        String::from(unsafe { CString::from_raw(ptr).to_str().unwrap() })
    }

    pub fn local_dead(&mut self, local: Local) {
        unsafe { __untraced_apitest_tracecompiler_local_dead(self.0, local) };
    }
}

impl Drop for TraceCompiler {
    fn drop(&mut self) {
        unsafe { __untraced_apitest_tracecompiler_drop(self.0) }
    }
}

impl SirTrace {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        unsafe { __untraced_apitest_sirtrace_len(self.0) }
    }
}

pub fn interpret_body<I>(body_name: &str, ctx: &mut I) {
    let body_cstr = CString::new(body_name).unwrap();
    unsafe { __untraced_apitest_interpret_body(body_cstr.as_ptr(), ctx as *mut _ as *mut u8) };
}

pub fn reg_pool_size() -> usize {
    unsafe { __untraced_apitest_reg_pool_size() }
}

pub fn compile_tir_trace<T>(mut tir_trace: TirTrace) -> Result<CompiledTrace<T>, CString> {
    let compiled = unsafe { __untraced_apitest_compile_tir_trace(tir_trace.0) };
    tir_trace.0 = ptr::null_mut(); // consumed.
    Ok(CompiledTrace {
        compiled,
        _marker: PhantomData,
    })
}

pub fn find_symbol(sym: &str) -> *mut c_void {
    let sym_cstr = CString::new(sym).unwrap();
    unsafe { __untraced_apitest_find_symbol(sym_cstr.as_ptr()) }
}
