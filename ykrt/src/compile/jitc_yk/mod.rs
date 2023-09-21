//! Yk's built-in trace compiler.

use crate::{
    compile::{CompiledTrace, Compiler},
    location::HotLocation,
    mt::{SideTraceInfo, MT},
    trace::MappedTrace,
};
use parking_lot::Mutex;
use std::{error::Error, ffi::CString, slice, sync::Arc};
use ykaddr::addr::symbol_vaddr;

mod aot_ir;

pub(crate) struct JITCYk;

impl Compiler for JITCYk {
    fn compile(
        &self,
        _mt: Arc<MT>,
        _irtrace: MappedTrace,
        sti: Option<SideTraceInfo>,
        _hl: Arc<Mutex<HotLocation>>,
    ) -> Result<CompiledTrace, Box<dyn Error>> {
        if sti.is_some() {
            todo!();
        }
        let ir_slice = yk_ir_section();
        let _aot_mod = aot_ir::deserialise_module(ir_slice);
        todo!();
    }

    #[cfg(feature = "yk_testing")]
    unsafe fn compile_for_tc_tests(
        &self,
        _irtrace: MappedTrace,
        _llvmbc_data: *const u8,
        _llvmbc_len: u64,
    ) {
        todo!()
    }
}

impl JITCYk {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {}))
    }
}

pub(crate) fn yk_ir_section() -> &'static [u8] {
    let start = symbol_vaddr(&CString::new("ykllvm.yk_ir.start").unwrap()).unwrap();
    let stop = symbol_vaddr(&CString::new("ykllvm.yk_ir.stop").unwrap()).unwrap();
    unsafe { slice::from_raw_parts(start as *const u8, stop - start) }
}
