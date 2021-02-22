//! The testing API client to ykshim.

use crate::{CompiledTrace, Local, RawCompiledTrace, RawSirTrace, RawTirTrace, SirTrace, TyIndex};
use libc::size_t;
use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::{fmt, ptr};

// Opaque pointers.
type RawTraceCompiler = c_void;

// Keep these types in-sync with the internal workspace.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct CguHash(u64);
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct TypeId {
    pub cgu: CguHash,
    pub idx: TyIndex,
}

extern "C" {
    fn __ykshimtest_compile_tir_trace(tir_trace: *mut RawTirTrace) -> *mut RawCompiledTrace;
    fn __ykshimtest_sirtrace_len(sir_trace: *mut RawSirTrace) -> size_t;
    fn __ykshimtest_tirtrace_new(sir_trace: *mut RawSirTrace) -> *mut RawTirTrace;
    fn __ykshim_tirtrace_drop(tir_trace: *mut RawTirTrace);
    fn __ykshimtest_tracecompiler_drop(comp: *mut RawTraceCompiler);
    fn __ykshimtest_tirtrace_len(tir_trace: *mut RawTirTrace) -> size_t;
    fn __ykshimtest_tirtrace_display(tir_trace: *mut RawTirTrace) -> *mut c_char;
    fn __ykshimtest_body_ret_ty(sym: *const c_char, ret_tyid: *mut TypeId);
    fn __ykshimtest_tracecompiler_default() -> *mut RawTraceCompiler;
    fn __ykshimtest_tracecompiler_insert_decl(
        tc: *mut RawTraceCompiler,
        local: Local,
        local_ty: TypeId,
        referenced: bool,
    );
    fn __ykshimtest_tracecompiler_local_to_location_str(
        tc: *mut RawTraceCompiler,
        local: Local,
    ) -> *mut c_char;
    fn __ykshimtest_tracecompiler_local_dead(tc: *mut RawTraceCompiler, local: Local);
    fn __ykshimtest_find_symbol(sym: *const c_char) -> *mut c_void;
    fn __ykshimtest_interpret_body(body_name: *const c_char, ctx: *mut u8);
    fn __ykshimtest_reg_pool_size() -> usize;
}

#[derive(Debug)]
pub struct TirTrace(*mut RawTirTrace);

impl TirTrace {
    pub fn new(sir_trace: &SirTrace) -> Self {
        Self(unsafe { __ykshimtest_tirtrace_new(sir_trace.0) })
    }

    pub fn len(&self) -> usize {
        unsafe { __ykshimtest_tirtrace_len(self.0) }
    }
}

impl Drop for TirTrace {
    fn drop(&mut self) {
        if self.0 != ptr::null_mut() {
            unsafe { __ykshim_tirtrace_drop(self.0) };
        }
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
    let mut ret = TypeId {
        cgu: CguHash(0),
        idx: TyIndex(0),
    };
    unsafe { __ykshimtest_body_ret_ty(sym_c.as_ptr(), &mut ret) };
    ret
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
            unsafe { __ykshimtest_tracecompiler_insert_decl(tc, *l, decl.ty, decl.referenced) };
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

pub fn interpret_body<I>(body_name: &str, ctx: &mut I) {
    let body_cstr = CString::new(body_name).unwrap();
    unsafe { __ykshimtest_interpret_body(body_cstr.as_ptr(), ctx as *mut _ as *mut u8) };
}

pub fn reg_pool_size() -> usize {
    unsafe { __ykshimtest_reg_pool_size() }
}

pub fn compile_tir_trace<T>(mut tir_trace: TirTrace) -> Result<CompiledTrace<T>, CString> {
    let compiled = unsafe { __ykshimtest_compile_tir_trace(tir_trace.0) };
    tir_trace.0 = ptr::null_mut(); // consumed.
    Ok(CompiledTrace {
        compiled,
        _marker: PhantomData,
    })
}

pub fn find_symbol(sym: &str) -> *mut c_void {
    let sym_cstr = CString::new(sym).unwrap();
    unsafe { __ykshimtest_find_symbol(sym_cstr.as_ptr()) }
}
