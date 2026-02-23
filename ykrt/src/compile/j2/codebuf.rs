//! Support for code buffers.
//!
//! When creating a code buffer, one starts with [CodeBufInProgress]: this allocates memory, but
//! does not require anything meaningful to have been written into it. When code has actually been
//! generated, [CodeBufInProgress::into_execodebuf] turns a [CodeBufInProgress] into a
//! [ExeCodeBuf]. An [ExeCodeBuf] is complete and executable as-is: although it might be patched,
//! the quantity of memory it contains will not change.

use crate::compile::j2::SyncSafePtr;
use libc::{__errno_location, PROT_EXEC, PROT_READ, PROT_WRITE, mprotect, munmap};
#[cfg(test)]
use libc::{MAP_ANON, MAP_FAILED, MAP_PRIVATE, mmap};
use parking_lot::Mutex;
use std::ffi::c_void;

/// A code buffer that does backing memory allocated but no actual code stored in it.
#[derive(Debug)]
pub(super) struct CodeBufInProgress {
    /// A pointer to the beginning of the `mmap`ed buffer.
    buf: *mut u8,
    /// How many bytes have we allocated to the buffer?
    len: usize,
}

impl CodeBufInProgress {
    /// Create a new code buffer with a size at least `len` bytes big.
    pub fn new(buf: *mut u8, len: usize) -> Self {
        Self { buf, len }
    }

    #[cfg(test)]
    pub fn new_testing() -> Self {
        let len = page_size::get();
        let buf = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_ANON | MAP_PRIVATE,
                -1,
                0,
            )
        };
        if buf == MAP_FAILED {
            todo!();
        }

        Self {
            buf: buf as *mut u8,
            len,
        }
    }

    /// Get a raw pointer to the start of the executable code buffer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.buf
    }

    /// Return the size of this buffer in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Copy `other_len` bytes from `other` into `self` and produce an [ExeCodeBuf].
    pub unsafe fn into_execodebuf(mut self, used: usize, entry: *mut u8) -> ExeCodeBuf {
        // If there's more than a page's worth of unused space in the buffer, return it to the OS,
        // and update `self.buf` and `self.len` appropriately.
        let unused = self.len - used.next_multiple_of(page_size::get());
        if unused > 0 {
            let rtn = unsafe { munmap(self.buf as *mut c_void, unused) };
            if rtn != 0 {
                todo!();
            }
            self.buf = unsafe { self.buf.byte_add(unused) };
            self.len -= unused;
        }

        // Remove write permissions.
        if unsafe { mprotect(self.buf as *mut c_void, self.len, PROT_EXEC | PROT_READ) } == -1 {
            todo!();
        }

        ExeCodeBuf {
            buf: SyncSafePtr(self.buf),
            start_off: self.len - used,
            len: self.len,
            entry: SyncSafePtr(entry),
            patch_lock: Mutex::new(()),
        }
    }
}

/// An executable code buffer.
#[derive(Debug)]
pub(super) struct ExeCodeBuf {
    /// A pointer to the beginning of the `mmap`ed buffer.
    buf: SyncSafePtr<*mut u8>,
    /// The offset of the start of the used part of [Self::buf].
    start_off: usize,
    /// How many bytes have we allocated to [Self::buf]?
    len: usize,
    /// A pointer to the executable entry point in this buffer.
    entry: SyncSafePtr<*mut u8>,
    /// This lock is used during patching: it is a simple way of ensuring that we don't race with
    /// another thread when `mprotect`ing page permissions. Note: this works because we assume that
    /// each [ExeCodeBuf] has allocated its own page. If we start sharing traces within pages, a
    /// semi-global lock will be needed.
    patch_lock: Mutex<()>,
}

impl ExeCodeBuf {
    /// Get a raw pointer to the start of the executable part of the code buffer.
    pub fn entry_ptr(&self) -> *const u8 {
        self.entry.0
    }

    /// Get a raw pointer to the start of the executable code buffer.
    pub fn sidetrace_entry(&self, sidetrace_off: usize) -> *const u8 {
        unsafe { self.buf.0.byte_add(self.start_off + sidetrace_off) }
    }

    /// Return the size of this buffer in bytes.
    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Patch part of the executable code. The address `patch_off...patch_off + len` bytes from the
    /// start of the buffer will be temporarily marked writable, at which point `f` will be called
    /// with the concrete address starting at `patch_off`. When `f` has completed, the executable
    /// buffer will have writable permissions removed.
    pub fn patch<F>(&self, patch_off: usize, patch_len: usize, f: F)
    where
        F: FnOnce(*mut u8),
    {
        let patch_ptr = unsafe { self.buf.0.byte_add(self.start_off + patch_off) };
        // mprotect requires a page-aligned address so round `patch_ptr` down.
        let page_ptr = ((patch_ptr.addr() / page_size::get()) * page_size::get()) as *mut u8;
        // `len` could span more than one page, so we need to account for that.
        let len =
            (patch_ptr.addr() + patch_len).next_multiple_of(page_size::get()) - page_ptr.addr();

        let _lk = self.patch_lock.lock();
        if unsafe {
            mprotect(
                page_ptr as *mut c_void,
                len,
                PROT_EXEC | PROT_READ | PROT_WRITE,
            )
        } == -1
        {
            todo!("{}", unsafe { *__errno_location() });
        }
        f(patch_ptr);
        if unsafe { mprotect(page_ptr as *mut c_void, len, PROT_EXEC | PROT_READ) } == -1 {
            todo!("{}", unsafe { *__errno_location() });
        }
    }
}
