//! An LLVM JIT backend. Currently a minimal wrapper around the fact that [IRTrace]s are hardcoded
//! to be compiled with LLVM.

use crate::{compile::Compiler, trace::IRTrace};
use libc::c_void;
use std::{error::Error, sync::Arc};
use tempfile::NamedTempFile;

pub(crate) struct JITCLLVM;

impl JITCLLVM {
    pub(crate) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(JITCLLVM))
    }
}

impl Compiler for JITCLLVM {
    fn compile(
        &self,
        irtrace: IRTrace,
    ) -> Result<(*const c_void, Option<NamedTempFile>), Box<dyn Error>> {
        irtrace.compile()
    }
}
