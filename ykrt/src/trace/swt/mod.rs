//! Software tracer.

use crate::frame::BitcodeSection;

use super::{errors::InvalidTraceError, AOTTraceIterator, TraceAction, TraceRecorder};
use std::sync::Once;
use std::{cell::RefCell, collections::HashMap, error::Error, ffi::CString, sync::Arc};

mod iterator;
use iterator::SWTraceIterator;

static FUNC_NAMES_INIT: Once = Once::new();

#[derive(Debug, Eq, PartialEq, Clone)]
struct TracingBlock {
    function_index: usize,
    block_index: usize,
}

thread_local! {
    // Collection of traced basicblocks.
    static BASIC_BLOCKS: RefCell<Vec<TracingBlock>> = RefCell::new(vec![]);
    // Mapping of function indexes to function names.
    static FUNC_NAMES: RefCell<HashMap<usize, CString>> = RefCell::new(HashMap::new());
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

/// Inserts LLVM IR basicblock metadata into a thread-local BASIC_BLOCKS vector.
///
/// # Arguments
/// * `function_index` - The index of the function to which the basic block belongs.
/// * `block_index` - The index of the basic block within the function.
pub fn trace_basicblock(function_index: usize, block_index: usize) {
    BASIC_BLOCKS.with(|v| {
        v.borrow_mut().push(TracingBlock {
            function_index,
            block_index,
        });
    })
}

pub(crate) struct SWTracer {}

impl SWTracer {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(SWTracer {})
    }
}

impl super::Tracer for SWTracer {
    fn start_recorder(self: Arc<Self>) -> Result<Box<dyn TraceRecorder>, Box<dyn Error>> {
        BASIC_BLOCKS.with(|bbs| {
            bbs.borrow_mut().clear();
        });
        Ok(Box::new(SWTTraceRecorder {}))
    }
}

struct SWTTraceRecorder {}

impl TraceRecorder for SWTTraceRecorder {
    fn stop(self: Box<Self>) -> Result<Box<dyn AOTTraceIterator>, InvalidTraceError> {
        let mut aot_blocks: Vec<TraceAction> = vec![];
        BASIC_BLOCKS.with(|tb| {
            FUNC_NAMES.with(|fnames| {
                FUNC_NAMES_INIT.call_once(|| {
                    let mut functions: *mut IRFunctionNameIndex = std::ptr::null_mut();
                    let bc_section = crate::compile::jitc_llvm::llvmbc_section();
                    let mut functions_len: usize = 0;
                    unsafe {
                        get_function_names(
                            &BitcodeSection {
                                data: bc_section.as_ptr(),
                                len: u64::try_from(bc_section.len()).unwrap(),
                            },
                            &mut functions,
                            &mut functions_len,
                        );
                        for entry in std::slice::from_raw_parts(functions, functions_len) {
                            fnames.borrow_mut().insert(
                                entry.index,
                                std::ffi::CStr::from_ptr(entry.name).to_owned(),
                            );
                        }
                    }
                });

                aot_blocks = tb
                    .borrow()
                    .iter()
                    .map(|tb| match fnames.borrow_mut().get(&tb.function_index) {
                        Some(name) => TraceAction::MappedAOTBlock {
                            func_name: name.to_owned(),
                            bb: tb.block_index,
                        },
                        _ => panic!(
                            "Failed to get function name by index {:?}",
                            tb.function_index
                        ),
                    })
                    .collect();
            })
        });
        if aot_blocks.is_empty() {
            Err(InvalidTraceError::EmptyTrace)
        } else {
            Ok(Box::new(SWTraceIterator::new(aot_blocks)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::Tracer;
    use crate::trace::{AOTTraceIterator, AOTTraceIteratorError, TraceAction};
    use std::sync::Arc;

    #[test]
    fn test_basic_blocks_content() {
        assert_eq!(BASIC_BLOCKS.with(|blocks| blocks.borrow().clone()), []);
        trace_basicblock(0, 0);
        trace_basicblock(0, 1);
        trace_basicblock(1, 0);
        assert_eq!(
            BASIC_BLOCKS.with(|blocks| blocks.borrow().clone()),
            vec![
                TracingBlock {
                    function_index: 0,
                    block_index: 0,
                },
                TracingBlock {
                    function_index: 0,
                    block_index: 1,
                },
                TracingBlock {
                    function_index: 1,
                    block_index: 0,
                },
            ]
        );
    }
}
