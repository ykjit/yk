//! yk's old trace compiler. Kept around for a few small bits.

use std::{error::Error, slice, sync::LazyLock};
use ykaddr::addr::symbol_to_ptr;

pub mod aot_ir;
pub(super) mod arbbitint;
mod int_signs;

pub(crate) static AOT_MOD: LazyLock<aot_ir::Module> = LazyLock::new(|| {
    let ir_slice = yk_ir_section().unwrap();
    aot_ir::deserialise_module(ir_slice).unwrap()
});

pub(crate) fn yk_ir_section() -> Result<&'static [u8], Box<dyn Error>> {
    let start = symbol_to_ptr("ykllvm.yk_ir.start")? as *const u8;
    let stop = symbol_to_ptr("ykllvm.yk_ir.stop")? as *const u8;
    debug_assert!(start < stop);
    Ok(unsafe { slice::from_raw_parts(start, stop.sub(start as usize) as usize) })
}
