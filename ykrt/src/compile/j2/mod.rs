//! The j2 trace compiler.
//!
//! This is a "reverse code generation" trace compiler. At a high-level it has three main passes:
//!
//! 1. Build a HIR trace from a sequence of AOT blocks ([aot_to_hir]) and optimise it as we go
//!    ([opt]).
//! 3. Assemble a HIR trace to machine code (using [hir_to_asm], [regalloc], and an
//!    architecture-dependent backend).
//!
//! Pass 1 is a a "forward" (i.e. normal) pass. Pass 2 is a "reverse" pass: roughly speaking, it
//! iterates from the last to the first instruction in a trace.

mod aot_to_hir;
mod codebuf;
mod compiled_trace;
mod hir;
#[cfg(test)]
mod hir_parser;
mod hir_to_asm;
mod opt;
mod regalloc;
#[cfg(target_arch = "x86_64")]
mod x64;

use crate::{
    compile::{
        CompilationError, CompiledTrace, Compiler, GuardId, TraceEndFrame,
        j2::codebuf::CodeBufInProgress, jitc_yk::AOT_MOD,
    },
    location::HotLocation,
    mt::{MT, TraceId},
    trace::AOTTraceIterator,
};
use libc::{MAP_ANON, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE, mmap, munmap};
use parking_lot::Mutex;
use std::{
    cell::RefCell,
    collections::HashMap,
    error::Error,
    ffi::{CString, c_void},
    sync::Arc,
};

thread_local! {
    /// A cache of dlsym results for this thread. The global dlsym cache is held in
    /// [J2::dlsym_cache].
    static THREAD_DLSYM_CACHE: RefCell<HashMap<String, Option<SyncSafePtr<*const c_void>>>> = RefCell::new(HashMap::new());
}

#[derive(Debug)]
pub(super) struct J2 {
    /// Cache non-thread-local dlsym() lookups.
    global_dlsym_cache: Mutex<HashMap<String, Option<SyncSafePtr<*const c_void>>>>,
    /// The address to pass to the next `mmap` call in the hope that it will place the block close
    /// enough in virtual memory that we we can use near calls. In normal execution, this starts
    /// with the address of `main`, although in testing mode (and perhaps other situations) the
    /// pointer can be null. As we receive values back from `mmap`, this hint will be updated.
    mmap_hint: Mutex<SyncSafePtr<*const c_void>>,
}

impl J2 {
    pub(super) fn new() -> Result<Arc<Self>, Box<dyn Error>> {
        Ok(Arc::new(Self {
            global_dlsym_cache: Mutex::new(HashMap::new()),
            #[cfg(not(test))]
            mmap_hint: Mutex::new(dlsym("main").unwrap()),
            #[cfg(test)]
            // When we're testing, no `main` function may be available.
            mmap_hint: Mutex::new(SyncSafePtr(std::ptr::null()))
        }))
    }

    /// Produce a new [CodeBufInProgress] of at least `len` bytes.
    ///
    /// This function will try to allocate the code buffer sufficiently near the interpreter and
    /// other traces that "near" jumps can be used. This cannot be guaranteed, especially as how
    /// far "near" is varies across platforms.
    fn mmap_codebufinprogress(&self, len: usize) -> CodeBufInProgress {
        let len = len.next_multiple_of(page_size::get());

        // Our aim below is to try and allocate memory "close" to existing code such that we can
        // generate near jumps. There is no guaranteed way to do this, so we use a similar idea to
        // LuaJIT's `mcode_alloc` function: we call `mmap` with a hint address and if we get back
        // memory that's too far away, we `munmap` it, update our hint address and try again. There
        // is, of course, no guarantee that this will succeed: it's particularly likely to fail
        // during the very first call, when we don't really know how close to `main` we can `mmap`
        // a page.
        let mut lk = self.mmap_hint.lock();
        let mut hint_ptr = lk.0;
        for _ in 0..16 {
            let buf = unsafe {
                mmap(
                    hint_ptr as *mut c_void,
                    len,
                    PROT_READ | PROT_WRITE,
                    MAP_ANON | MAP_PRIVATE,
                    -1,
                    0,
                ) as *const c_void
            };
            if buf == MAP_FAILED {
                todo!();
            }

            let is_near = if !hint_ptr.is_null() {
                #[cfg(target_arch = "x86_64")]
                let is_near = unsafe { hint_ptr.offset_from(buf) }.abs() < 0x80000000;

                is_near
            } else {
                // If `hint_ptr` is null -- e.g. during testing -- we give up and say "this'll do".
                true
            };

            if is_near {
                *lk = SyncSafePtr(unsafe { buf.byte_add(len) });
                return CodeBufInProgress::new(buf as *mut u8, len);
            }

            if unsafe { munmap(buf as *mut c_void, len) } == -1 {
                todo!();
            }

            // Should we use a different heuristic for the first call? When `hint_ptr` is `main`'s
            // address, it's likely that we'll have to go around this loop several times at least,
            // but subsequent calls are much more likely to give us what we hope for on the first
            // iteration.
            hint_ptr = unsafe { hint_ptr.byte_add(64 * 1024) };
        }

        todo!();
    }

    /// Convert a symbol name to an address. This is a caching front-end to [libc::dlsym].
    /// `is_thread_local` must be set to `true` if `symbol` is a thread-local, or an undefined
    /// address will be returned.
    fn dlsym(&self, symbol: &str, is_thread_local: bool) -> Option<SyncSafePtr<*const c_void>> {
        THREAD_DLSYM_CACHE.with_borrow_mut(|b| {
            if let Some(x) = b.get(symbol) {
                // The optimal case: we have cached `symbol` in this thread's cache.
                return *x;
            }

            // Lookup `symbol` either as a thread-local or in the global cache.
            let ptr = if is_thread_local {
                dlsym(symbol)
            } else {
                // If `symbol` is not thread-local we can try the global cache, inserting a value
                // for `symbol` if it's not already present.
                *self
                    .global_dlsym_cache
                    .lock()
                    .entry(symbol.to_owned())
                    .or_insert_with(|| dlsym(symbol))
            };

            // Cache `symbol` in this thread.
            b.insert(symbol.to_owned(), ptr);
            ptr
        })
    }
}

impl Compiler for J2 {
    fn root_compile(
        self: Arc<Self>,
        mt: Arc<MT>,
        ta_iter: Box<dyn AOTTraceIterator>,
        trid: TraceId,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        coupler: Option<std::sync::Arc<dyn CompiledTrace>>,
        _endframe: TraceEndFrame,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        let kind = match coupler {
            Some(tgt_ctr) => aot_to_hir::BuildKind::Coupler { tgt_ctr },
            None => aot_to_hir::BuildKind::Loop,
        };

        #[cfg(target_arch = "x86_64")]
        type AotToHir = aot_to_hir::AotToHir<x64::Reg>;

        let hm = AotToHir::new(
            &mt,
            &self,
            &AOT_MOD,
            Arc::clone(&hl),
            ta_iter,
            trid,
            kind,
            promotions,
            debug_strs,
        )
        .build()?;

        #[cfg(target_arch = "x86_64")]
        let minlen = x64::x64hir_to_asm::X64HirToAsm::codebuf_minlen(&hm);
        let buf = self.mmap_codebufinprogress(minlen);
        #[cfg(target_arch = "x86_64")]
        let be = x64::x64hir_to_asm::X64HirToAsm::new(&hm, buf);

        hir_to_asm::HirToAsm::new(&hm, hl, be).build(mt)
    }

    fn sidetrace_compile(
        self: Arc<Self>,
        mt: Arc<MT>,
        ta_iter: Box<dyn AOTTraceIterator>,
        trid: TraceId,
        src_ctr: Arc<dyn CompiledTrace>,
        src_gid: GuardId,
        tgt_ctr: Arc<dyn CompiledTrace>,
        hl: Arc<Mutex<HotLocation>>,
        promotions: Box<[u8]>,
        debug_strs: Vec<String>,
        _endframe: TraceEndFrame,
    ) -> Result<Arc<dyn CompiledTrace>, CompilationError> {
        #[cfg(target_arch = "x86_64")]
        type AotToHir = aot_to_hir::AotToHir<x64::Reg>;

        let hm = AotToHir::new(
            &mt,
            &self,
            &AOT_MOD,
            Arc::clone(&hl),
            ta_iter,
            trid,
            aot_to_hir::BuildKind::Side {
                src_ctr,
                src_gid,
                tgt_ctr,
            },
            promotions,
            debug_strs,
        )
        .build()?;

        #[cfg(target_arch = "x86_64")]
        let minlen = x64::x64hir_to_asm::X64HirToAsm::codebuf_minlen(&hm);
        let buf = self.mmap_codebufinprogress(minlen);
        #[cfg(target_arch = "x86_64")]
        let be = x64::x64hir_to_asm::X64HirToAsm::new(&hm, buf);

        hir_to_asm::HirToAsm::new(&hm, hl, be).build(mt)
    }
}

#[derive(Clone, Copy, Debug)]
struct SyncSafePtr<T>(T);
unsafe impl<T> Send for SyncSafePtr<T> {}
unsafe impl<T> Sync for SyncSafePtr<T> {}

/// A non-caching wrapper around `dlsym`.
fn dlsym(symbol: &str) -> Option<SyncSafePtr<*const c_void>> {
    let cn = CString::new(symbol).unwrap();
    let ptr = unsafe { libc::dlsym(std::ptr::null_mut(), cn.as_c_str().as_ptr()) } as *const c_void;
    if ptr.is_null() {
        None
    } else {
        Some(SyncSafePtr(ptr))
    }
}
