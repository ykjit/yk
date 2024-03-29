//! Software tracer.

use super::{
    AOTTraceIterator, AOTTraceIteratorError, TraceAction, TraceRecorder, TraceRecorderError, Tracer,
};
use crate::{
    compile::jitc_llvm::frame::BitcodeSection,
    mt::{MTThread, DEFAULT_TRACE_TOO_LONG},
};
use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error,
    ffi::CString,
    sync::{Arc, LazyLock},
};

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBlock {
    function_index: usize,
    block_index: usize,
}

// Mapping of function indexes to function names.
static FUNC_NAMES: LazyLock<HashMap<usize, CString>> = LazyLock::new(|| {
    let mut fnames = HashMap::new();
    let mut functions: *mut IRFunctionNameIndex = std::ptr::null_mut();
    let bc_section = crate::compile::jitc_llvm::llvmbc_section();
    let mut functions_len: usize = 0;
    let bs = &BitcodeSection {
        data: bc_section.as_ptr(),
        len: u64::try_from(bc_section.len()).unwrap(),
    };
    unsafe { get_function_names(bs, &mut functions, &mut functions_len) };
    for entry in unsafe { std::slice::from_raw_parts(functions, functions_len) } {
        fnames.insert(
            entry.index,
            unsafe { std::ffi::CStr::from_ptr(entry.name) }.to_owned(),
        );
    }
    fnames
});

thread_local! {
    // Collection of traced basicblocks.
    static BASIC_BLOCKS: RefCell<Vec<TracingBlock>> = RefCell::new(vec![]);
}

/// Inserts LLVM IR basicblock metadata into a thread-local BASIC_BLOCKS vector.
///
/// # Arguments
/// * `function_index` - The index of the function to which the basic block belongs.
/// * `block_index` - The index of the basic block within the function.
#[cfg(tracer_swt)]
#[no_mangle]
pub extern "C" fn yk_trace_basicblock(function_index: usize, block_index: usize) {
    MTThread::with(|mtt| {
        if mtt.is_tracing() {
            BASIC_BLOCKS.with(|v| {
                v.borrow_mut().push(TracingBlock {
                    function_index,
                    block_index,
                });
            })
        }
    });
}

extern "C" {
    fn get_function_names(
        section: *const BitcodeSection,
        result: *mut *mut IRFunctionNameIndex,
        len: *mut usize,
    );
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct IRFunctionNameIndex {
    pub index: usize,
    pub name: *const libc::c_char,
}

pub(crate) struct SWTracer {}

impl SWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(SWTracer {})
    }
}

impl Tracer for SWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        debug_assert!(BASIC_BLOCKS.with(|bbs| bbs.borrow().is_empty()));
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, TraceRecorderError> {
        let bbs = BASIC_BLOCKS.with(|tb| tb.replace(Vec::new()));
        if bbs.len() > DEFAULT_TRACE_TOO_LONG {
            Err(TraceRecorderError::TraceTooLong)
        } else if bbs.is_empty() {
            // FIXME: who should handle an empty trace?
            panic!();
        } else {
            Ok(Box::new(SWTraceIterator::new(bbs)))
        }
    }
}

struct SWTraceIterator {
    bbs: std::vec::IntoIter<TracingBlock>,
}

impl SWTraceIterator {
    fn new(bbs: Vec<TracingBlock>) -> SWTraceIterator {
        return SWTraceIterator {
            bbs: bbs.into_iter(),
        };
    }
}

impl Iterator for SWTraceIterator {
    type Item = Result<TraceAction, AOTTraceIteratorError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.bbs
            .next()
            .map(|tb| match FUNC_NAMES.get(&tb.function_index) {
                Some(name) => Ok(TraceAction::MappedAOTBlock {
                    func_name: name.to_owned(),
                    bb: tb.block_index,
                }),
                _ => panic!(
                    "Failed to get function name by index {:?}",
                    tb.function_index
                ),
            })
    }
}

impl AOTTraceIterator for SWTraceIterator {}
