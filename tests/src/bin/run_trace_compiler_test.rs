//! Trace compiler test runner.
//!
//! Each invocation of this program runs one of the trace compiler tests found in the
//! `trace_driver` directory of this crate.

use memmap2;
use std::{collections::HashMap, env, ffi::CString, fs::File};
use yktrace::{IRBlock, IRTrace};

fn main() {
    // Build the trace that we are going to have compiled.
    let mut bbs = vec![IRBlock::unmappable()];
    for bb in env::var("TRACE_DRIVER_BBS").unwrap().split(",") {
        let mut elems = bb.split(":");
        let func = elems.next().unwrap();
        let bb_idx = elems.next().unwrap().parse::<usize>().unwrap();
        bbs.push(IRBlock::new(CString::new(func).unwrap(), bb_idx));
    }
    let trace = IRTrace::new(bbs, HashMap::new());

    // Map the `.ll` file into the address space so that we can give a pointer to it to the trace
    // compiler. Normally (i.e. outside of testing), the trace compiler wouldn't deal with textual
    // bitcode format, but it just so happens that LLVM's module loading APIs accept either format.
    let ll_path = env::args().skip(1).next().unwrap();
    let ll_file = File::open(ll_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&ll_file).unwrap() };

    trace.compile_for_tc_tests(mmap.as_ptr(), mmap.len());
}
