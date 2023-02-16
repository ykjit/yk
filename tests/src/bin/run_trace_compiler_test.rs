//! Trace compiler test runner.
//!
//! Each invocation of this program runs one of the trace compiler tests found in the
//! `trace_compiler` directory of this crate.

use memmap2;
use std::{collections::HashMap, env, error::Error, ffi::CString, fs::File};
use yktrace::{IRBlock, IRTrace};

const BBS_ENV: &str = "YKT_TRACE_BBS";

fn parse_bb(bb: &str) -> Result<(CString, usize), Box<dyn Error>> {
    let mut elems = bb.split(':');
    let func = elems.next().ok_or("malformed function name")?;
    let bb_idx = elems
        .next()
        .ok_or("malformed basic block index")?
        .parse::<usize>()?;
    Ok((CString::new(func)?, bb_idx))
}

fn main() -> Result<(), String> {
    // Build the trace that we are going to have compiled.
    let mut bbs = vec![];
    if let Ok(tbbs) = env::var(BBS_ENV) {
        for bb in tbbs.split(',') {
            if let Ok((func, bb_idx)) = parse_bb(bb) {
                bbs.push(IRBlock::new(func, bb_idx));
            } else {
                return Err(format!("{} is malformed", BBS_ENV));
            }
        }
    } else {
        return Err(format!(
            "The test doesn't set the {} environment variable",
            BBS_ENV
        ));
    }
    let trace = IRTrace::new(bbs, HashMap::new());

    // Map the `.ll` file into the address space so that we can give a pointer to it to the trace
    // compiler. Normally (i.e. outside of testing), the trace compiler wouldn't deal with textual
    // bitcode format, but it just so happens that LLVM's module loading APIs accept either format.
    let ll_path = env::args().nth(1).unwrap();
    let ll_file = File::open(ll_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&ll_file).unwrap() };

    unsafe { trace.compile_for_tc_tests(mmap.as_ptr(), mmap.len()) };

    Ok(())
}
